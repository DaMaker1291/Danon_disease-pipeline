"""The paired-seed PFS extension, reported as its OWN estimand.

``outputs_real/outcome_cards.json`` calls ``pfs_extension`` the RATIO OF THE TWO
ARMS' MEDIAN PFS -- ``median_policy / median_mtd - 1``. ``danon/campaign.py``'s
own rule (check S1) describes a different quantity: the PFS extension over
PAIRED seeds, reported as a bootstrap 95% interval over replicate PAIRS rather
than as a bare number. The two are not interchangeable, and presenting one under
the other's name is exactly the drift ``danon/claims.py`` exists to catch, so
this module carries both and labels them.

It runs no engine: the replicates are already on disk in
``outputs_real/campaign_log.json``, so the interval is reproducible offline from
the same bytes every time. The bootstrap seed is fixed for that reason.

What it CANNOT do is turn a simulator comparison into an observed outcome. PFS
here is the simulator's own RANO >= 25% rule, and the adaptive arm reaches the
365-day horizon in every recorded replicate -- so every paired difference is a
right-censored LOWER BOUND, never a point estimate, and the endpoint is a
property of the model rather than of a patient.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The recorded campaign log: paired per-replicate PFS, the only input here.
CAMPAIGN_LOG_PATH = os.path.join(ROOT, "outputs_real", "campaign_log.json")

#: Fixed so the recorded interval is reproducible byte-for-byte on any machine.
PAIRED_EXTENSION_SEED = 20261008
PAIRED_EXTENSION_N_BOOT = 2000


def load_campaign_log(path: Optional[str] = None) -> List[Dict[str, object]]:
    """The recorded campaign log, as a list of per-case entries."""
    with open(path or CAMPAIGN_LOG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def paired_pfs_extension_pairs(
    entries: List[Dict[str, object]]
) -> Tuple[List[float], List[float], List[Dict[str, object]]]:
    """Every PAIRED replicate difference, pooled across the recorded cases.

    Returns ``(diffs_days, ratios, per_case)`` where ``diffs_days[r]`` is
    ``policy - mtd`` in days and ``ratios[r]`` is ``policy/mtd - 1`` for the same
    pair. ``per_case`` records the same pairing case by case, beside the
    ratio-of-medians value the campaign log already carries.
    """
    diffs: List[float] = []
    ratios: List[float] = []
    per_case: List[Dict[str, object]] = []

    for entry in entries:
        policy = [float(v) for v in (entry.get("pfs_policy_replicates") or [])]
        mtd = [float(v) for v in (entry.get("pfs_mtd_replicates") or [])]
        n = min(len(policy), len(mtd))

        case_diffs: List[float] = []
        case_ratios: List[float] = []
        for i in range(n):
            p, m = policy[i], mtd[i]
            case_diffs.append(p - m)
            case_ratios.append((p / m - 1.0) if m else float("nan"))

        diffs.extend(case_diffs)
        ratios.extend(case_ratios)
        per_case.append(
            {
                "case_id": entry.get("case_id"),
                "patient_id": entry.get("patient_id"),
                "n_pairs": n,
                "median_pfs_policy": entry.get("policy_median_pfs"),
                "median_pfs_mtd": entry.get("paired_mtd_median_pfs"),
                # The outcome cards' statistic, carried beside the paired one.
                "ratio_of_medians_extension": entry.get("pfs_extension_median"),
                "paired_mean_difference_days": (
                    float(np.mean(case_diffs)) if case_diffs else None
                ),
            }
        )

    return diffs, ratios, per_case


def paired_pfs_extension_report(
    entries: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """The paired-seed extension, its bootstrap interval, and its censoring.

    Two statistics are carried and labelled: the paired replicate difference in
    DAYS (with a bootstrap 95% interval over pairs, the repository's S1 rule),
    and the per-case RATIO OF MEDIANS the outcome cards call ``pfs_extension``.
    """
    from danon.campaign import bootstrap_ci_mean

    log = entries if entries is not None else load_campaign_log()
    diffs, _ratios, per_case = paired_pfs_extension_pairs(log)

    horizon = 365
    n_pairs = len(diffs)
    mean_days = float(np.mean(diffs)) if diffs else None
    median_days = float(np.median(diffs)) if diffs else None
    ci = bootstrap_ci_mean(
        diffs, n_boot=PAIRED_EXTENSION_N_BOOT, seed=PAIRED_EXTENSION_SEED
    )

    # Right-censoring: an adaptive replicate sitting on the horizon is a lower
    # bound, so a difference computed from it is one too.
    at_horizon = sum(
        1
        for entry in log
        for v in (entry.get("pfs_policy_replicates") or [])
        if float(v) >= horizon
    )

    ratio_values = [
        float(c["ratio_of_medians_extension"])
        for c in per_case
        if c["ratio_of_medians_extension"] is not None
    ]

    return {
        "n_cases": len(log),
        "n_pairs": n_pairs,
        "horizon_days": horizon,
        "bootstrap_n_boot": PAIRED_EXTENSION_N_BOOT,
        "bootstrap_seed": PAIRED_EXTENSION_SEED,
        "paired_difference_estimand": (
            "mean of (policy PFS - MTD PFS) in days, over replicates paired by seed"
        ),
        "paired_mean_difference_days": mean_days,
        "paired_median_difference_days": median_days,
        "paired_mean_difference_ci95_days": (list(ci) if ci else None),
        "n_policy_replicates_at_horizon": at_horizon,
        "is_right_censored_lower_bound": bool(at_horizon > 0),
        "ratio_of_medians_estimand": (
            "median_policy / median_mtd - 1, per case (the outcome cards' "
            "`pfs_extension` label)"
        ),
        "ratio_of_medians_extension_median": (
            float(np.median(ratio_values)) if ratio_values else None
        ),
        "records": per_case,
        "estimand_note": (
            "two statistics are carried and they are NOT interchangeable: the "
            "paired replicate difference in DAYS, and the per-case RATIO OF "
            "MEDIANS the outcome cards call `pfs_extension`. Quoting one under "
            "the other's name is the drift this record is meant to prevent."
        ),
        "lower_bound_note": (
            f"right-censored: {at_horizon} adaptive replicate(s) reached the "
            f"{horizon} d horizon, so every paired difference is a LOWER BOUND "
            "and the interval is a bound on a bound -- never a bare point "
            "estimate of a difference."
        ),
        "limit": (
            "a simulator comparison over paired seeds on model geometry: PFS is "
            "the simulator's own RANO >= 25% rule, not observed patient "
            "progression-free survival, and no patient-level validation exists "
            "here."
        ),
    }
