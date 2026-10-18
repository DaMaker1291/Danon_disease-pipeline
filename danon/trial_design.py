"""A trial-design screen: the counterfactual stated, the noise shown, schedules ruled out.

A design brief asks for three things that a simulator CAN honestly do, and
this module derives all three from records that already exist:

  * a COUNTERFACTUAL: what the model did under each arm, said as a statement
    about the math and its declared assumptions rather than as a prediction
    about a patient,
  * a NOISE FLOOR: the paired-seed differences with a resampled interval, so a
    screen can say out loud when two arms' spreads overlap and the simulated
    difference is therefore UNRESOLVABLE at that replicate count -- and, where
    the record carries too few pairs to form the repository's own percentile
    interval at all, that it is UNASSESSED rather than resolved,
  * a RULE-OUT: the calendars a DECLARED screen rejects -- and nothing more.

WHY THE RULE-OUT IS THE HONEST HALF OF A PRESCRIPTION. ``danon.dosing``'s
``MyelosuppressionModel`` is a DECLARED screen: its delay kernel is fitted to
the label's single reported datum (the median day of neutrophil nadir), drug B's
marrow potency is an ASSUMED number, the baseline count is a typical adult
reference rather than this patient's, and the class docstring states it is not
clinical decision support. So a calendar it rejects is rejected BY THE MODEL,
which is a true statement about this tree, and is NOT a regulatory finding --
the label's 1.5 x 10^9/L withhold threshold is applied through a declared
convolution, not read off a licence. That asymmetry is the whole point: ruling a
schedule OUT only claims the model refused it, while naming a schedule to GIVE
claims something about a patient that no code here can establish.

It runs no engine: the replicates are on disk in
``outputs_real/outcome_cards.json``, so every number re-derives offline from the
same bytes. What it cannot do is turn a simulator comparison into an observed
outcome -- the endpoint is the model's own RANO >= 25% burden rule, the adaptive
arm reaches the 365-day horizon in every recorded replicate (so each paired
difference is a right-censored lower bound), and no patient-level validation
exists in this tree.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The recorded outcome cohort -- the same artifact the browser reads, so the
#: screen and the record are over identical bytes.
OUTCOME_CARDS_PATH = os.path.join(ROOT, "outputs_real", "outcome_cards.json")

#: Fixed so the recorded fanning intervals are reproducible on any machine.
TRIAL_DESIGN_SEED = 20261008
#: Confidence levels the panel fans out at, narrowest first.
TRIAL_DESIGN_LEVELS: Tuple[float, ...] = (0.50, 0.80, 0.95)

#: The explicit label a screen shows when two arms' spreads overlap and the
#: simulated difference cannot be separated at this replicate count.
NOISE_FLOOR_FLAG = "[SIMULATED BENEFIT WITHIN NOISE FLOOR - UNRESOLVABLE]"

#: The label for the OTHER way a set of replicates fails to resolve: too few
#: pairs to form the repository's own percentile interval at all. It is a
#: different statement from the one above and gets a different label, because a
#: set of three all-positive pairs is not "within a noise floor" -- it is
#: unassessed, and calling those the same thing would be its own overclaim.
INSUFFICIENT_REPLICATES_FLAG = "[NO INTERVAL AT THIS REPLICATE COUNT - UNRESOLVED]"

#: Below this many pairs the percentile bootstrap refuses to answer, so the
#: rules above are not a choice made here but a property of the method.
MIN_PAIRS_FOR_AN_INTERVAL = 5

#: The label a pool carries when its pairs are NESTED rather than independent. A
#: bootstrap over paired replicates treats them as exchangeable, and in this
#: record they are not: three seeds share one configuration, and three
#: configurations share one tumour geometry. The count and the nesting are
#: disclosed beside the count, never left to be inferred from it.
NESTED_REPLICATES_FLAG = "[NESTED REPLICATES - NOT INDEPENDENT SAMPLES]"


def _nesting_note(
    subject: str,
    assumed: int,
    configurations: int,
    geometries: int,
    seeds: int,
    units: int,
) -> str:
    """The one nesting disclosure, rendered for whichever pool it describes.

    Mirrored word for word in ``frontend/src/utils/trialDesign.ts``. A pool
    over a single geometry names that one image; the whole-record pool names
    them all. Keeping one template means the two scales can differ only in the
    counts they interpolate, never in what they say those counts mean.
    """
    scope = (
        f"{configurations} configurations across {geometries} imaging geometries"
        if geometries > 1
        else f"{configurations} configurations on a single image"
    )
    return (
        f"{subject} resamples {assumed} paired replicates as exchangeable, but "
        f"they come from {scope}, {seeds} seeds each: seeds within a "
        "configuration are the same model under a different seed rather than a "
        "new patient, and configurations sharing a geometry are the same tumour "
        "under different synthetic initial conditions, so the interval rests on "
        f"at most {units} nested units and is narrower than one over independent "
        "patients would be"
    )


def _nesting_counts(cases: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """The counts a nesting disclosure states, derived from per-case rows.

    A bootstrap over paired replicates assumes they are exchangeable. In this
    record they are NESTED on two axes -- seeds within a configuration, and
    configurations within one imaging geometry -- so the count of pairs is not a
    count of independent samples. These are the numbers the disclosure is about,
    computed once here so both scales count the same rows the same way.
    """
    n_pairs = sum(int(c.get("n_pairs") or 0) for c in cases)
    seeds = [int(c.get("n_seeds") or 0) for c in cases]

    per_geometry: Dict[str, int] = {}
    configurations = set()
    for case in cases:
        geometry = str(case.get("case_id"))
        per_geometry[geometry] = per_geometry.get(geometry, 0) + 1
        configurations.add((geometry, case.get("r0")))

    return {
        "n_pairs": n_pairs,
        # What the bootstrap treats as exchangeable -- the number the interval is
        # computed over, stated so it is never confused with the units below.
        "assumed_exchangeable": n_pairs,
        "n_configurations": len(cases),
        "independent_units": len(configurations),
        "n_geometries": len(per_geometry),
        "configurations_per_geometry": max(per_geometry.values()) if per_geometry else 0,
        "seeds_per_configuration": max(seeds) if seeds else 0,
    }


def _pool_nesting(cases: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """The disclosure for one image pool: every pair on ONE tumour geometry."""
    counts = _nesting_counts(cases)
    return {
        **counts,
        "note": _nesting_note(
            "this pool's interval",
            counts["assumed_exchangeable"],
            counts["n_configurations"],
            counts["n_geometries"],
            counts["seeds_per_configuration"],
            counts["independent_units"],
        ),
    }


def _record_nesting(cases: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """The disclosure for the whole record: the pairs across every geometry."""
    counts = _nesting_counts(cases)
    return {
        **counts,
        "note": _nesting_note(
            "the pooled interval",
            counts["assumed_exchangeable"],
            counts["n_configurations"],
            counts["n_geometries"],
            counts["seeds_per_configuration"],
            counts["independent_units"],
        ),
    }


def load_outcome_cards(path: Optional[str] = None) -> Dict[str, object]:
    """The recorded outcome cohort, as its own artifact carries it."""
    with open(path or OUTCOME_CARDS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _arm_pfs(arm: Dict[str, object], horizon: int) -> List[float]:
    """Per-replicate PFS in days, with a non-progressing replicate set to the horizon.

    ``pfs_day`` is ``None`` when a replicate did not progress inside the
    horizon; the value it contributes is the horizon itself, which makes the
    resulting difference a LOWER BOUND for that pair -- flagged, never quietly
    treated as a measurement.
    """
    out: List[float] = []
    for r in arm.get("replicate_day_readouts") or []:
        day = r.get("pfs_day")
        out.append(float(horizon) if day is None else float(day))
    return out


def _inversion_day(series: Sequence[float], threshold: float) -> Optional[int]:
    """First 1-based day the simulated resistant fraction reaches ``threshold``.

    The screen calls this the simulated INVERSION day: it is the day the
    resistant clone stops being the minority, which is the moment the arm's
    own switch rule acts on.
    """
    for i, v in enumerate(series):
        if float(v) >= threshold:
            return i + 1
    return None


def _switch_threshold(switch_rule: str) -> float:
    """The resistant-fraction threshold read out of the recorded switch rule."""
    import re

    m = re.search(r"resistant fraction\s*>=\s*([0-9.]+)\s*%", switch_rule or "")
    if not m:
        return 0.15
    return float(m.group(1)) / 100.0


def paired_seed_differences(
    cards: Dict[str, object],
) -> Tuple[List[float], List[Dict[str, object]]]:
    """Every paired replicate difference, pooled and case by case.

    Returns ``(pooled_diffs, per_case)`` where ``pooled_diffs[r]`` is
    ``policy PFS - MTD PFS`` in days for a replicate paired BY POSITION (the
    two arms are recorded with the same seeds in the same order).
    """
    horizon = int(cards.get("horizon_days") or 365)
    threshold = _switch_threshold(str(cards.get("switch_rule") or ""))
    pooled: List[float] = []
    per_case: List[Dict[str, object]] = []

    for card in cards.get("cases") or []:
        arms = card.get("arms") or {}
        pol = _arm_pfs(arms.get("policy") or {}, horizon)
        mtd = _arm_pfs(arms.get("mtd") or {}, horizon)
        n = min(len(pol), len(mtd))
        diffs = [pol[i] - mtd[i] for i in range(n)]
        pooled.extend(diffs)

        daily_p = (arms.get("policy") or {}).get("daily") or {}
        daily_m = (arms.get("mtd") or {}).get("daily") or {}
        inv_p = _inversion_day(
            daily_p.get("resistant_fraction") or [], threshold
        )
        inv_m = _inversion_day(
            daily_m.get("resistant_fraction") or [], threshold
        )

        n_censored_policy = sum(
            1
            for r in (arms.get("policy") or {}).get("replicate_day_readouts") or []
            if r.get("pfs_day") is None
        )
        head = card.get("day_365") or {}

        per_case.append(
            {
                "case_id": card.get("case_id"),
                "patient_id": card.get("patient_id"),
                # The synthetic initial resistant fraction. Each case is a
                # CONFIGURATION of one geometry, not a patient; the r0 sweep is
                # what the nesting disclosure is about.
                "r0": (card.get("intake") or {}).get("r0"),
                "n_seeds": max(len(pol), len(mtd)),
                "n_pairs": n,
                "diffs_days": diffs,
                "n_ties": sum(1 for d in diffs if d == 0.0),
                "n_violations": sum(1 for d in diffs if d < 0),
                "n_censored_policy_replicates": n_censored_policy,
                "is_right_censored_lower_bound": bool(n_censored_policy > 0),
                "median_pfs_policy_days": head.get("median_pfs_policy"),
                "median_pfs_mtd_days": head.get("median_pfs_mtd"),
                "ratio_of_medians_extension": head.get("pfs_extension"),
                "ratio_of_medians_label": head.get("pfs_extension_label"),
                "inversion_day_policy": inv_p,
                "inversion_day_mtd": inv_m,
                "inversion_delay_days": (
                    None if inv_p is None or inv_m is None else inv_p - inv_m
                ),
                "switch_threshold": threshold,
            }
        )

    return pooled, per_case


def _fan(diffs: Sequence[float]) -> Dict[str, object]:
    """A resampled interval over the paired differences, fanned by confidence.

    Uses the repository's own percentile bootstrap over the MEAN
    (``danon.campaign.bootstrap_ci_mean``, the S1 rule's method) at each level
    in ``TRIAL_DESIGN_LEVELS`` on a fixed seed, so the recorded interval is
    reproducible offline. Returns ``None`` for a level when fewer than five
    pairs exist, because a percentile interval over three pairs is a decoration
    rather than an estimate.
    """
    from danon.campaign import bootstrap_ci_mean

    intervals: Dict[str, Optional[List[float]]] = {}
    for i, level in enumerate(TRIAL_DESIGN_LEVELS):
        ci = bootstrap_ci_mean(
            list(diffs),
            n_boot=2000,
            seed=TRIAL_DESIGN_SEED + i,
            level=level,
        )
        intervals[f"ci{int(round(level * 100))}"] = list(ci) if ci else None
    return intervals


def _noise_floor(
    diffs: Sequence[float], intervals: Dict[str, Optional[List[float]]]
) -> Dict[str, object]:
    """Whether the simulated difference can be separated from zero at all.

    Three states, and they are different from one another. RESOLVES means the
    widest interval excludes zero and at least one pair separated the arms.
    WITHIN_NOISE_FLOOR means the widest interval contains zero and gets
    ``NOISE_FLOOR_FLAG``. NO_INTERVAL_AT_THIS_REPLICATE_COUNT means there were
    fewer pairs than the method requires, so nothing was estimated at all and
    the distinct ``INSUFFICIENT_REPLICATES_FLAG`` says so -- a handful of
    all-positive pairs is not a benefit inside a noise floor, and calling the
    two the same thing would be its own overclaim. Tie and violation counts are
    reported beside the state rather than folded into it, because a set whose
    interval excludes zero can still contain replicates that failed to separate
    -- both arms sitting on the horizon -- and hiding those inside an aggregate
    is how a spread gets printed as a point.
    """
    ci_wide = intervals.get("ci95")
    spans_zero = bool(ci_wide is not None and ci_wide[0] <= 0.0 <= ci_wide[1])
    n_ties = sum(1 for d in diffs if d == 0.0)
    n_violations = sum(1 for d in diffs if d < 0)
    n_separated = sum(1 for d in diffs if d > 0.0)
    no_pair_separated = bool(diffs) and n_separated == 0

    if ci_wide is None:
        status = "no_interval_at_this_replicate_count"
        flag = INSUFFICIENT_REPLICATES_FLAG
        reason = (
            f"{len(diffs)} paired replicate(s) is below the "
            f"{MIN_PAIRS_FOR_AN_INTERVAL} the repository's own percentile "
            "method requires, so no interval exists here and the difference "
            "is unassessed rather than resolvable"
        )
    elif spans_zero or no_pair_separated:
        status = "within_noise_floor"
        flag = NOISE_FLOOR_FLAG
        reason = (
            "the widest resampled interval contains zero, so the two arms' "
            "spreads overlap and the simulated difference is not resolvable at "
            "this replicate count"
            if spans_zero
            else f"not one of the {len(diffs)} paired replicate(s) separated "
            "the arms, so there is no resolvable difference to report"
        )
    else:
        status = "resolves"
        flag = None
        reason = (
            f"the widest interval excludes zero and {n_separated} of "
            f"{len(diffs)} paired replicate(s) separated the arms; "
            f"{n_ties} tied and {n_violations} went the other way"
        )

    return {
        "status": status,
        "ci_spans_zero": spans_zero,
        "n_pairs": len(diffs),
        "n_ties": n_ties,
        "n_violations": n_violations,
        "n_separated": n_separated,
        "no_pair_separated": no_pair_separated,
        "unresolvable_at_this_n": bool(flag is not None),
        "reason": reason,
        "flag": flag,
    }


def _candidate_schedules() -> List[Tuple[str, object, str]]:
    """The declared calendars the rule-out panel evaluates, and why each is here.

    Every entry is a DECLARED calendar built from ``danon.dosing``'s own
    primitives -- no search, no optimizer, nothing chosen because it looked
    good. Items 1-2 are the label's regimen and the brief's own arithmetic
    (5 on / 14 off), items 3-5 probe the dose and duration axes, and items 6-7
    are the combination shapes the screen's own docstring names.
    """
    from danon.dosing import LABEL_CYCLE_DAYS, LABEL_DOSE_MG_M2, LABEL_ON_DAYS, Schedule

    h = 365
    return [
        (
            "label_regimen_150x5_of_28",
            Schedule.cyclic("A", LABEL_DOSE_MG_M2, LABEL_ON_DAYS, LABEL_CYCLE_DAYS, h),
            "the FDA label's own regimen: 150 mg/m2, days 1-5 of each 28-day "
            "cycle. This is the calendar the screen's own budget is DEFINED by, "
            "so it is the reference row, not a proposal",
        ),
        (
            "brief_arithmetic_150x5_of_19",
            Schedule.cyclic("A", LABEL_DOSE_MG_M2, LABEL_ON_DAYS, 19, h),
            "the brief's own arithmetic -- 5 days on, then a 14-day holiday -- "
            "which is a 19-day cycle, not the label's 28",
        ),
        (
            "label_plus_one_day_150x6_of_28",
            Schedule.cyclic("A", LABEL_DOSE_MG_M2, LABEL_ON_DAYS + 1, LABEL_CYCLE_DAYS, h),
            "one extra dose-day per cycle at the label dose: the smallest "
            "duration escalation the screen can be asked about",
        ),
        (
            "continuous_150_every_day",
            Schedule.continuous("A", LABEL_DOSE_MG_M2, h),
            "continuous dosing at the label's daily dose -- the standard-care "
            "arm the model is compared AGAINST, written as an unbroken calendar",
        ),
        (
            "escalated_200x5_of_28",
            Schedule.cyclic("A", 200.0, LABEL_ON_DAYS, LABEL_CYCLE_DAYS, h),
            "the same calendar at 200 mg/m2/day, to probe the dose axis",
        ),
        (
            "alternating_AB_150_block5_rest23",
            Schedule.alternating(150.0, 150.0, 5, h, rest_days=23),
            "A then B, 5 days each on a 28-day cycle: the only combination "
            "shape the screen permits at the label's own mg/m2",
        ),
        (
            "alternating_AB_150_block5_rest0",
            Schedule.alternating(150.0, 150.0, 5, h, rest_days=0),
            "A then B back to back, no holiday: a continuous two-drug calendar",
        ),
    ]


def declared_anc_ruleout(horizon_days: int = 365) -> Dict[str, object]:
    """Evaluate a DECLARED set of calendars against the DECLARED ANC screen.

    Nothing here is a search and nothing here is a regulatory determination.
    ``MyelosuppressionModel`` is the screen; its kernel is fitted to one label
    datum, drug B's marrow potency is an ASSUMED weight swept rather than set,
    and its own docstring states it is not clinical decision support. A row that
    reads ``feasible: False`` therefore means THE MODEL REFUSED THIS CALENDAR --
    a true statement about this tree -- and a reader must not read it as a
    licence decision.
    """
    from danon.dosing import MyelosuppressionModel

    model = MyelosuppressionModel()
    rows: List[Dict[str, object]] = []
    for label, schedule, note in _candidate_schedules():
        rows.append(
            {
                "label": label,
                "note": note,
                "feasible": bool(model.feasible(schedule, horizon_days)),
                "nadir_anc": float(model.nadir(schedule, horizon_days)),
                "peak_exposure_over_budget": (
                    float(
                        model.marrow_exposure(schedule, horizon_days).max()
                        / model.suppression_budget()
                    )
                    if model.suppression_budget() > 0
                    else None
                ),
                "assumed_potency_drugs": list(model.unknown_drug_potency(schedule)),
            }
        )

    rejected = [r["label"] for r in rows if not r["feasible"]]
    return {
        "declared": True,
        "is_a_regulatory_boundary": False,
        "screen": "danon.dosing.MyelosuppressionModel",
        "screen_status": (
            "a DECLARED screen, not a licence: its kernel is fitted to the "
            "label's single reported datum (the median day of neutrophil "
            "nadir), drug B's marrow potency is an ASSUMED weight, the "
            "baseline count is a typical adult reference rather than this "
            "patient's, and its own docstring states it is not clinical "
            "decision support"
        ),
        "budget_continuous_mg_m2": float(model.max_continuous_mg_m2()),
        "anchors": model.anchor_checks(horizon_days),
        "candidates": rows,
        "n_rejected": len(rejected),
        "rejected": rejected,
        "rule_out_note": (
            "a rejected row means the MODEL refused this calendar under its "
            "declared parameters. That is a statement about the math and is "
            "safe to publish as one; it is NOT a statement that a calendar is "
            "clinically unsafe, and the reverse -- naming a calendar to give -- "
            "is not something this screen can do at all"
        ),
    }


def _group_intervals(
    per_case: Sequence[Dict[str, object]], key: str
) -> List[Dict[str, object]]:
    """Pool the paired differences by ``key`` and report each pool's summary.

    Grouping is how a resolvability ladder falls out of the record without
    inventing a sample: the same pairs, viewed at the per-case scale, at the
    per-geometry scale, and pooled. Each pool owns its own summary -- its
    interval, its resolution state and its PFS medians -- so a consumer renders
    what the pool says rather than re-deriving it, which is what keeps the
    browser's live groups and this record the same shape.
    """
    grouped: "OrderedDict[str, List[Dict[str, object]]]" = OrderedDict()
    for case in per_case:
        grouped.setdefault(str(case.get(key)), []).append(case)

    out: List[Dict[str, object]] = []
    for name, members in grouped.items():
        diffs = [d for case in members for d in case["diffs_days"]]
        intervals = _fan(diffs)
        # Each case's own median PFS, then the median across the group's cases.
        policy = [
            float(c["median_pfs_policy_days"])
            for c in members
            if c.get("median_pfs_policy_days") is not None
        ]
        mtd = [
            float(c["median_pfs_mtd_days"])
            for c in members
            if c.get("median_pfs_mtd_days") is not None
        ]
        out.append(
            {
                key: name,
                "n_pairs": len(diffs),
                "mean_difference_days": float(np.mean(diffs)) if diffs else None,
                "median_pfs_policy_days": (
                    float(np.median(policy)) if policy else None
                ),
                "median_pfs_mtd_days": (float(np.median(mtd)) if mtd else None),
                "diffs_days": diffs,
                "intervals": intervals,
                "noise_floor": _noise_floor(diffs, intervals),
                "nesting": _pool_nesting(members),
            }
        )
    return out


def trial_design_report(cards: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """The counterfactual, the noise floor and the rule-out, all from records."""
    cards = cards if cards is not None else load_outcome_cards()
    horizon = int(cards.get("horizon_days") or 365)

    pooled, per_case = paired_seed_differences(cards)
    pooled_intervals = _fan(pooled)
    pooled_floor = _noise_floor(pooled, pooled_intervals)
    pooled_nesting = _record_nesting(per_case)

    for case in per_case:
        intervals = _fan(case["diffs_days"])
        case["intervals"] = intervals
        case["noise_floor"] = _noise_floor(case["diffs_days"], intervals)

    by_geometry = _group_intervals(per_case, "case_id")
    n_case_flags = sum(
        1 for c in per_case if (c["noise_floor"] or {}).get("unresolvable_at_this_n")
    )
    n_geom_flags = sum(
        1 for g in by_geometry if (g["noise_floor"] or {}).get("unresolvable_at_this_n")
    )
    delays = [
        c["inversion_delay_days"]
        for c in per_case
        if c["inversion_delay_days"] is not None
    ]
    n_equal = sum(1 for d in delays if d == 0)

    return {
        "source": "outputs_real/outcome_cards.json (recorded paired replicates)",
        "n_cases": len(per_case),
        "n_pairs": len(pooled),
        "horizon_days": horizon,
        "bootstrap_seed": TRIAL_DESIGN_SEED,
        "levels": list(TRIAL_DESIGN_LEVELS),
        "paired_difference_estimand": (
            "policy PFS - MTD PFS in days, over replicates paired by position "
            "within a case"
        ),
        "pooled": {
            "mean_difference_days": float(np.mean(pooled)) if pooled else None,
            "median_difference_days": float(np.median(pooled)) if pooled else None,
            "intervals": pooled_intervals,
            "noise_floor": pooled_floor,
            "nesting": pooled_nesting,
        },
        "cases": per_case,
        "by_geometry": by_geometry,
        "n_cases_unresolvable": n_case_flags,
        "n_geometries_unresolvable": n_geom_flags,
        "resolvability": {
            "definition": (
                "the same paired differences viewed at three scales -- within "
                "one case, pooled over one image's cases, and pooled over the "
                "whole record -- because whether a simulated difference is "
                "resolvable is a property of the replicate count, not of the "
                "comparison"
            ),
            "min_pairs_for_an_interval": MIN_PAIRS_FOR_AN_INTERVAL,
            "ladder": [
                {
                    "scale": "within one case",
                    "groups": len(per_case),
                    "pairs_per_group": (
                        per_case[0]["n_pairs"] if per_case else 0
                    ),
                    "groups_without_a_resolvable_interval": n_case_flags,
                },
                {
                    "scale": "pooled over one image",
                    "groups": len(by_geometry),
                    "pairs_per_group": (
                        by_geometry[0]["n_pairs"] if by_geometry else 0
                    ),
                    "groups_without_a_resolvable_interval": n_geom_flags,
                },
                {
                    "scale": "pooled over the whole record",
                    "groups": 1,
                    "pairs_per_group": len(pooled),
                    "groups_without_a_resolvable_interval": (
                        1 if pooled_floor["unresolvable_at_this_n"] else 0
                    ),
                },
            ],
            # A count-derived template, mirrored word for word in
            # frontend/src/utils/trialDesign.ts, so the live screen and this
            # record derive the same sentence from the same counts and a drift
            # between them is visible on the screen rather than silent.
            "note": (
                "a pool without a resolvable interval is either below the "
                f"{MIN_PAIRS_FOR_AN_INTERVAL} paired replicates the method needs, or an "
                "interval that still contains zero: "
                f"{n_case_flags} of {len(per_case)} single-case pools, "
                f"{n_geom_flags} of {len(by_geometry)} image pools, and "
                f"{1 if pooled_floor['unresolvable_at_this_n'] else 0} pooled "
                "whole-record row(s) are in that state"
            ),
        },
        "inversion": {
            "definition": (
                "the first 1-based day the simulated resistant fraction reaches "
                "the arm's own switch threshold"
            ),
            "n_cases_with_a_delay": sum(1 for d in delays if d != 0),
            "n_cases_with_equal_inversion_day": n_equal,
            "delay_note": (
                "in the recorded replicates the two arms reach the switch "
                "threshold on the SAME simulated day in every case, so the "
                "model shows no simulated delay in this inversion -- the arms "
                "diverge after it, in burden, not in when it happens"
                if delays and n_equal == len(delays)
                else "the recorded replicates show a non-zero simulated delay "
                "in the inversion day for at least one case; read the per-case "
                "rows rather than a summary"
            ),
        },
        "anc_screen": declared_anc_ruleout(horizon),
        "noise_floor_flag_text": NOISE_FLOOR_FLAG,
        "insufficient_replicates_flag_text": INSUFFICIENT_REPLICATES_FLAG,
        "nested_replicates_flag_text": NESTED_REPLICATES_FLAG,
        "what_this_screen_is": [
            "a statement about what the model DID under declared assumptions: "
            "two arms, the same recorded geometry and paired seeds, the model's "
            "own RANO >= 25% burden rule as the endpoint",
            "a rule-out: calendars the DECLARED ANC screen refuses, named as "
            "such. Ruling a schedule out claims the model rejected it and "
            "nothing about a patient",
        ],
        "what_this_screen_is_not": [
            (
                "a sample of independent patients: the "
                f"{pooled_nesting['n_pairs']} paired replicates are nested -- "
                f"{pooled_nesting['seeds_per_configuration']} seeds on each of "
                f"{pooled_nesting['n_configurations']} configurations, and the "
                f"{pooled_nesting['n_configurations']} configurations are "
                f"{pooled_nesting['configurations_per_geometry']} synthetic "
                "initial conditions on each of "
                f"{pooled_nesting['n_geometries']} imaging geometries -- so the "
                f"interval treats {pooled_nesting['assumed_exchangeable']} pairs "
                f"as exchangeable when at most "
                f"{pooled_nesting['independent_units']} nested units stand "
                "behind them"
            ),
            "a schedule to give anyone: naming a calendar to administer is a "
            "clinical claim this repository cannot make and does not make",
            "an observed outcome: every replicate is a simulation, the adaptive "
            "arm is censored at the horizon, and no patient-level validation "
            "exists in this tree",
            "a regulatory determination: the ANC screen is DECLARED, its kernel "
            "is fitted to a single label datum, and drug B's marrow potency is "
            "assumed",
        ],
    }


def trial_design_provenance() -> Dict[str, EvidenceStatus]:
    """The honesty envelope for this screen."""
    return {
        "engine_arithmetic": EvidenceStatus.ESTABLISHED,
        "comparison_is_over_paired_replicates": EvidenceStatus.ESTABLISHED,
        "anc_screen_parameters_are_measured": EvidenceStatus.PROVEN_FAILED,
        "engine_predicts_patients": EvidenceStatus.UNKNOWN,
        "clinical_benefit": EvidenceStatus.UNKNOWN,
    }


def what_a_trial_design_screen_cannot_do() -> List[str]:
    return [
        "turn a simulator comparison into an observed progression: an interval "
        "over paired model replicates bounds a model's arithmetic, not a "
        "patient's course",
        "resolve arms whose spreads overlap: when the widest interval contains "
        "zero or any replicate ties, the difference is unresolvable at this "
        "replicate count, and reporting a point estimate instead is invention",
        "promote a rule-out to a safety finding: a rejected calendar was refused "
        "by a DECLARED screen whose kernel is fitted to one label datum and "
        "whose drug-B potency is assumed",
        "select a schedule to administer: that is a clinical decision this "
        "repository cannot support, and the benefit badge stays UNKNOWN until "
        "prospective trials and regulatory review",
    ]


__all__ = [
    "INSUFFICIENT_REPLICATES_FLAG",
    "MIN_PAIRS_FOR_AN_INTERVAL",
    "NESTED_REPLICATES_FLAG",
    "NOISE_FLOOR_FLAG",
    "TRIAL_DESIGN_LEVELS",
    "TRIAL_DESIGN_SEED",
    "declared_anc_ruleout",
    "load_outcome_cards",
    "paired_seed_differences",
    "trial_design_provenance",
    "trial_design_report",
    "what_a_trial_design_screen_cannot_do",
]
