"""Real dosing calendars, a real safety constraint, and a real search.

WHAT THIS MODULE IS
-------------------
A calendar schedule ("150 mg/m2 for 5 days, then a 21-day holiday, repeat")
is just a deterministic function of the day index. The engine already asks its
policy for ``(dose_A, dose_B)`` once per day, so an open-loop calendar is
expressed exactly as a `danon.evosteer.BasePolicy`. That is all this module is:
the calendar, a safety constraint over it, and a search that runs the real
engine to rank candidates.

Every number this module returns is produced by running
`danon.evosteer.TumorEvolutionSimulator`. Nothing is interpolated, sketched,
illustrated or back-filled. If a number is not measured here it is reported as
an assumption, and `provenance()` says which is which.

WHAT IS ASSUMED, AND MUST STAY LABELLED AS SUCH
------------------------------------------------
1. `MyelosuppressionModel`. The safety screen is ANCHORED TO THE TEMOZOLOMIDE
   PRESCRIBING INFORMATION, not invented here: the ANC threshold it screens
   against is the label's own quoted "withhold until ANC >= 1.5 x 10^9/L", and
   the marrow budget is defined as the burden produced by the label's own
   regimen (150 mg/m2 on days 1-5 of each 28-day cycle). See
   `MyelosuppressionModel.anchor_checks`, which MEASURES that the screen
   reproduces that regimen, rejects a six-day block, puts the neutrophil nadir
   on the label's reported day, and stays non-cumulative across cycles.

   Two things remain declared choices, not measurements:
     * the marrow-suppression KERNEL (its shape) is a two-parameter log-normal
       fitted to the label's one reported nadir DAY; the depth of suppression
       is not fitted at all, it is fixed by the label regimen and threshold;
     * the circulating count is taken as proportional to instantaneous marrow
       output -- a quasi-steady simplification, declared because the
       suppression is sustained for weeks rather than delivered as impulses.

   No neutrophil count in this repository was measured on a patient, and drug
   B's marrow potency is ASSUMED (swept, never set to zero to manufacture a
   feasible calendar). This module implements a screen; it does not confer
   clinical safety and it is not a dosing rule.
2. Everything the engine carries about GBM drug response (see
   `evosteer.what_this_model_cannot_know`). Ranking schedules under an assumed
   biology ranks them under the assumption.

The single most important honesty rule here: `search_schedules` may only
report a winner that beat its baselines on the seeds it actually ran. If
nothing beats the baselines, it says so. It does not return the best candidate
it happened to evaluate and call it a win.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import (
    Callable, Dict, Iterable, List, Optional, Sequence, Tuple,
)

import numpy as np

from danon.evosteer import (
    BasePolicy,
    PolicyKind,
    PolicyState,
    TrajectorySpec,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)


# ---------------------------------------------------------------------------
# Two performance defaults, both CHOSEN BY MEASUREMENT on this machine
# ---------------------------------------------------------------------------

# Replicates per `run_batch` call. `measure_batch_scaling` derives it: at 24^3
# with float64 fields a batched replicate's per-call arrays stop fitting in
# cache and the bandwidth cost swamps the numpy call-overhead saving, so the
# fastest batch is the smallest one. It is 1, which is the opposite of what the
# batching was built on -- which is why it is measured rather than assumed, and
# why callers on other hardware should re-measure rather than inherit it.
MEASURED_BATCH_OPTIMUM = 1

# Worker processes for candidate-level parallelism, chosen the same way.
# `measure_worker_scaling` re-derives it. Measured here at 24^3 with 365-day
# trajectories: the cost per candidate falls only ~1.6x from 1 worker to 4 and
# then flattens, so the pool saturates at 4 and buys nothing beyond that --
# not the near-linear speedup a process pool advertises. Every worker count
# returned bit-identical medians. The absolute seconds are in the artifact's
# `worker_scaling` section; they move with machine load, the shape does not.
MEASURED_WORKER_OPTIMUM = 4


# ---------------------------------------------------------------------------
# The calendar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseBlock:
    """One uninterrupted block of one drug.

    `dose_level` is the ENGINE's normalised intensity in [0, 1], not a dose in
    mg/m2. `dose_mg_m2` is the clinical face of the same block and is only
    used by the myelosuppression model; the engine never sees it, because the
    engine has no mg/m2 scale.
    """

    drug: str                      # "A" or "B"
    start_day: int
    duration_days: int
    dose_level: float = 1.0        # engine intensity, [0, 1]
    dose_mg_m2: float = 150.0      # clinical face, mg/m2/day

    def __post_init__(self) -> None:
        if self.drug not in ("A", "B"):
            raise ValueError(f"drug must be 'A' or 'B', got {self.drug!r}")
        if self.start_day < 0:
            raise ValueError(f"start_day must be >= 0, got {self.start_day}")
        if self.duration_days <= 0:
            raise ValueError(f"duration_days must be > 0, got {self.duration_days}")
        if not 0.0 <= self.dose_level <= 1.0:
            raise ValueError(f"dose_level must be in [0, 1], got {self.dose_level}")
        if self.dose_mg_m2 < 0.0:
            raise ValueError(f"dose_mg_m2 must be >= 0, got {self.dose_mg_m2}")

    def active_on(self, day: int) -> bool:
        return self.start_day <= day < self.start_day + self.duration_days

    def to_dict(self) -> Dict:
        return {
            "drug": self.drug,
            "start_day": self.start_day,
            "duration_days": self.duration_days,
            "dose_level": self.dose_level,
            "dose_mg_m2": self.dose_mg_m2,
        }

    def label(self) -> str:
        return (
            f"{self.drug} {self.dose_mg_m2:g} mg/m2/d d{self.start_day}"
            f"-{self.start_day + self.duration_days - 1}"
        )


@dataclass(frozen=True)
class Schedule:
    """An ordered, non-overlapping set of blocks on one drug's calendar.

    Two blocks on the SAME drug may not overlap: an overlapping calendar has
    no single answer to "what was the dose on day d", and silently picking one
    would make the schedule's text disagree with what the engine ran. Blocks on
    DIFFERENT drugs may overlap freely -- that is combination therapy, not an
    ambiguity.
    """

    blocks: Tuple[DoseBlock, ...] = ()

    def __post_init__(self) -> None:
        by_drug: Dict[str, List[DoseBlock]] = {}
        for b in self.blocks:
            by_drug.setdefault(b.drug, []).append(b)
        for drug, bs in by_drug.items():
            ordered = sorted(bs, key=lambda b: b.start_day)
            for prev, nxt in zip(ordered, ordered[1:]):
                if nxt.start_day < prev.start_day + prev.duration_days:
                    raise ValueError(
                        f"overlapping {drug} blocks: {prev.label()} runs into "
                        f"{nxt.label()}; a same-drug calendar must be unambiguous"
                    )

    @staticmethod
    def cyclic(
        drug: str,
        dose_mg_m2: float,
        on_days: int,
        cycle_days: int,
        horizon_days: int,
        dose_level: float = 1.0,
        start_day: int = 0,
    ) -> "Schedule":
        """The pulsed schedule: `on_days` on, then rest until the cycle repeats.

        "150 mg/m2 for 5 days, 21-day holiday" is on_days=5, cycle_days=26.
        """
        if cycle_days <= 0 or on_days <= 0:
            raise ValueError("on_days and cycle_days must be > 0")
        if on_days > cycle_days:
            raise ValueError(
                f"on_days ({on_days}) cannot exceed cycle_days ({cycle_days})"
            )
        blocks: List[DoseBlock] = []
        start = start_day
        while start < horizon_days:
            blocks.append(
                DoseBlock(drug, start, on_days, dose_level, dose_mg_m2)
            )
            start += cycle_days
        return Schedule(tuple(blocks))

    @staticmethod
    def continuous(
        drug: str,
        dose_mg_m2: float,
        horizon_days: int,
        dose_level: float = 1.0,
        start_day: int = 0,
    ) -> "Schedule":
        return Schedule(
            (DoseBlock(drug, start_day, max(1, horizon_days - start_day),
                       dose_level, dose_mg_m2),)
        )

    @staticmethod
    def alternating(
        dose_a_mg_m2: float,
        dose_b_mg_m2: float,
        block_days: int,
        horizon_days: int,
        dose_level_a: float = 1.0,
        dose_level_b: float = 1.0,
        start_with: str = "A",
        start_day: int = 0,
        rest_days: int = 0,
    ) -> "Schedule":
        """Two drugs in strict alternation: A for `block_days`, then B, repeat.

        `rest_days` inserts an off-drug holiday after every block. With
        rest_days=0 this is back-to-back alternation (a continuous two-drug
        calendar); with rest_days >= block_days it is alternation on a cycle
        timescale -- "A for 5 days, off 23, B for 5 days, off 23" -- which is
        the only shape this screen permits at the label's own mg/m2, and the
        shape a real combination protocol would be written as.

        Deterministic and open-loop, which is the point: the same calendar runs
        the same way on every replicate, so a difference between this and a
        single-drug calendar is the schedule and not the controller.

        Distinguish this from the engine's own `CollateralSwitchPolicy`, which
        alternates on a MEASURED resistant fraction with hysteresis. This one
        alternates on the calendar. Both are legitimate; conflating them would
        attribute a closed-loop result to an open-loop rule.
        """
        if block_days <= 0:
            raise ValueError(f"block_days must be > 0, got {block_days}")
        if rest_days < 0:
            raise ValueError(f"rest_days must be >= 0, got {rest_days}")
        if start_with not in ("A", "B"):
            raise ValueError(f"start_with must be 'A' or 'B', got {start_with!r}")
        order = ("A", "B") if start_with == "A" else ("B", "A")
        doses = {"A": dose_a_mg_m2, "B": dose_b_mg_m2}
        levels = {"A": dose_level_a, "B": dose_level_b}
        blocks: List[DoseBlock] = []
        cursor = start_day
        turn = 0
        while cursor < horizon_days:
            drug = order[turn % 2]
            blocks.append(
                DoseBlock(drug, cursor, block_days, levels[drug], doses[drug])
            )
            cursor += block_days + rest_days
            turn += 1
        return Schedule(tuple(blocks))

    @staticmethod
    def alternating_relative(
        drug_b_dose_level: float,
        block_days: int,
        horizon_days: int,
        dose_mg_m2: float = 150.0,
        dose_level_a: float = 1.0,
        start_with: str = "A",
        start_day: int = 0,
        rest_days: int = 0,
    ) -> "Schedule":
        """Alternation expressed in the ENGINE's relative potency for drug B.

        Drug A keeps a fixed clinical face (`dose_mg_m2`); drug B is given as a
        level on A's scale. This is what makes the comparison interpretable:
        the engine has an absolute `kill_max_S` per drug, and asking "what
        level of the second drug" is answerable in the engine's own units
        without pretending to know drug B's mg/m2.
        """
        return Schedule.alternating(
            dose_mg_m2, dose_mg_m2, block_days, horizon_days,
            dose_level_a=dose_level_a, dose_level_b=drug_b_dose_level,
            start_with=start_with, start_day=start_day, rest_days=rest_days,
        )

    def dose_on(self, day: int) -> Tuple[float, float]:
        """(dose_A, dose_B) in [0, 1] on `day` -- the engine's own convention."""
        a = b = 0.0
        for blk in self.blocks:
            if blk.active_on(day):
                if blk.drug == "A":
                    a = max(a, blk.dose_level)
                else:
                    b = max(b, blk.dose_level)
        return a, b

    def daily_mg_m2(self, day: int) -> float:
        return sum(
            blk.dose_mg_m2 * blk.dose_level
            for blk in self.blocks if blk.active_on(day)
        )

    def drugs_used(self) -> Tuple[str, ...]:
        return tuple(sorted({b.drug for b in self.blocks}))

    def is_combination(self) -> bool:
        return len(self.drugs_used()) > 1

    def daily_exposure(
        self, day: int, weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Per-day marrow exposure in mg/m2, optionally weighted per drug.

        The engine's `dose_level` SCALES the exposure. A block written as
        "150 mg/m2 at 0.4 x full" is 60 mg/m2/day of drug, and a screen that
        read it as 150 mg/m2 would disagree with the calendar's own text --
        which `describe()` deliberately prints whenever the level is not 1.0.
        Getting this wrong once let a whole family of reduced-intensity
        calendars be screened as if they were full dose.

        A combination must not be screened as if both drugs were the same
        agent: the model's potency is calibrated for one drug's mg/m2, so a
        second agent's contribution is a stated multiplier, not a fact. The
        default weight of 1.0 is the *conservative* choice -- it assumes the
        second drug is fully myelosuppressive -- and `unknown_drug_potency`
        records that this is an assumption rather than a measurement.
        """
        w = weights or {}
        return sum(
            blk.dose_mg_m2 * blk.dose_level * float(w.get(blk.drug, 1.0))
            for blk in self.blocks if blk.active_on(day)
        )

    def total_mg_m2(self) -> float:
        return sum(
            blk.dose_mg_m2 * blk.dose_level * blk.duration_days
            for blk in self.blocks
        )

    def on_fraction(self, horizon_days: Optional[int] = None) -> float:
        """Share of the horizon the patient is actually on drug.

        The final block of a calendar built to a horizon is usually truncated by
        it, so both the numerator and the denominator are clipped to the same
        horizon. Counting a block's full nominal length against a horizon it
        runs past would overstate exposure at the end of the year.
        """
        if not self.blocks:
            return 0.0
        horizon = horizon_days
        if horizon is None:
            horizon = max(b.start_day + b.duration_days for b in self.blocks)
        horizon = max(1, int(horizon))
        on_days = 0
        for b in self.blocks:
            start = max(0, b.start_day)
            end = min(horizon, b.start_day + b.duration_days)
            on_days += max(0, end - start)
        return min(1.0, on_days / float(horizon))

    def dose_level_on(self, day: int) -> float:
        return max(self.dose_on(day))

    def to_dict(self, horizon_days: Optional[int] = None) -> Dict:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "total_mg_m2": self.total_mg_m2(),
            "on_fraction": self.on_fraction(horizon_days),
            "text": self.describe(),
        }

    def describe(self) -> str:
        if not self.blocks:
            return "no drug"
        by_drug: Dict[str, List[DoseBlock]] = {}
        for b in sorted(self.blocks, key=lambda b: (b.drug, b.start_day)):
            by_drug.setdefault(b.drug, []).append(b)
        parts = []
        for drug, bs in by_drug.items():
            first = bs[0]
            period = (bs[1].start_day - bs[0].start_day) if len(bs) > 1 else None
            # The engine's relative level is reported whenever it is not 1.0,
            # so a calendar cannot be described in mg/m2 alone and then run at
            # a different intensity than the text implies.
            lvl = (
                "" if abs(first.dose_level - 1.0) < 1e-12
                else f" at {first.dose_level:g} x full"
            )
            if period and len(bs) > 1:
                parts.append(
                    f"{drug}: {first.dose_mg_m2:g} mg/m2/d{lvl} x{first.duration_days} d, "
                    f"every {period} d ({len(bs)} blocks)"
                )
            else:
                parts.append(
                    f"{drug}: {first.dose_mg_m2:g} mg/m2/d{lvl} "
                    f"x{first.duration_days} d (single block)"
                )
        return "alternating: " + " | ".join(parts) if self.is_combination() else "; ".join(parts)


class SchedulePolicy(BasePolicy):
    """The calendar, as a real engine policy.

    IDENTITY: a schedule built from the same blocks the text describes produces
    exactly the dose series the text claims, because both come from
    `Schedule.dose_on`. `tests/test_dosing.py` pins that against the engine's
    own recorded `dose_A` / `dose_B` arrays.
    """

    kind = PolicyKind.MTD  # a calendar is open-loop; MTD is the closest base

    def __init__(self, schedule: Schedule, name: Optional[str] = None) -> None:
        self.schedule = schedule
        self._name = name or schedule.describe()

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        return self.schedule.dose_on(state.day)

    def reset(self) -> None:
        """A calendar holds no state between replicates."""

    def describe(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Safety constraint -- anchored to the FDA label, residual choices declared
# ---------------------------------------------------------------------------

TEMOZOLOMIDE_LABEL_URL = (
    "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
    "?setid=021b97cc-a76f-44b6-b7be-670d2ae83690"
)

# Transcribed from that label. Each entry is a threshold or a datum the screen
# is anchored to, so they sit next to the code that consumes them and a reader
# can check them against the source.
LABEL_QUOTES: Dict[str, Tuple[str, str]] = {
    "withhold_threshold": (
        "Prior to dosing, withhold TEMOZOLOMIDE until patients have an "
        "absolute neutrophil count (ANC) of 1.5 x 10^9/L or greater and a "
        "platelet count of 100 x 10^9/L or greater.",
        "DOSAGE AND ADMINISTRATION 2.1",
    ),
    "maintenance_withhold": (
        "Withhold TEMOZOLOMIDE if ANC less than 1 x 10^9/L.",
        "DOSAGE MODIFICATIONS DURING MAINTENANCE TREATMENT, Table 2",
    ),
    "regimen": (
        "Cycle 1: 150 mg/m2 per day on days 1 to 5",
        "RECOMMENDED DOSAGE AND DOSAGE MODIFICATIONS 2.2",
    ),
    "regimen_cycle": (
        "The recommended initial dosage of TEMOZOLOMIDE is 150 mg/m2 once "
        "daily on Days 1 to 5 of each 28-day cycle.",
        "DOSAGE AND ADMINISTRATION",
    ),
    "nadir_day": (
        "the median nadirs occurred at 26 days for platelets (range: 21 to 40 "
        "days) and 28 days for neutrophils (range: 1 to 44 days)",
        "WARNINGS AND PRECAUTIONS 5.1, study MK-7365-006",
    ),
    "not_cumulative": (
        "myelosuppression usually occurred during the first few cycles of "
        "therapy and was generally not cumulative",
        "WARNINGS AND PRECAUTIONS 5.1",
    ),
}

LABEL_ANC_FLOOR = 1.5      # 10^9/L -- LABEL_QUOTES["withhold_threshold"]
LABEL_DOSE_MG_M2 = 150.0   # LABEL_QUOTES["regimen"]
LABEL_ON_DAYS = 5          # LABEL_QUOTES["regimen"]
LABEL_CYCLE_DAYS = 28      # LABEL_QUOTES["regimen_cycle"]
LABEL_NADIR_DAY = 28       # 1-based -- LABEL_QUOTES["nadir_day"]


@dataclass(frozen=True)
class MyelosuppressionModel:
    """A neutrophil trajectory driven by cumulative marrow exposure.

    THE SHAPE OF THE ARITHMETIC. Every dose-day loads a marrow-suppression
    reservoir. The reservoir is the dose series convolved with a log-normal
    delay kernel -- so suppression builds over days, peaks about three weeks
    later, and then clears -- and the circulating ANC is taken as the baseline
    scaled by the surviving marrow output:

        S(t)   = sum over t' of w(t - t') * marrow_weighted_dose(t')
        f(t)   = max(0, 1 - (1 - floor/baseline) * S(t) / budget)
        ANC(t) = baseline * f(t)

    The kernel is normalised to unit area, so a CONSTANT marrow-weighted dose
    of `d` mg/m2/day settles at a reservoir of `d`, and the budget reads
    directly as the highest continuous marrow burden this screen permits.

    THE BUDGET IS DEFINED, NOT CHOSEN: `suppression_budget` is the reservoir
    produced by the LABEL'S OWN regimen -- 150 mg/m2 on days 1-5 of each
    28-day cycle. The screen's boundary therefore lands on the regimen the
    label prescribes rather than on a number picked to make a search work,
    and `anchor_checks` MEASURES that it does.

    DECLARED, NOT MEASURED (see the module docstring):
      * `kernel_log_mu` / `kernel_log_sd` shape the kernel. They are fitted to
        the label's single reported datum -- the median day of neutrophil nadir
        -- and to nothing else. A different plausible kernel would move the
        feasible frontier, and that is stated here rather than buried.
      * ANC proportional to marrow output is a quasi-steady simplification,
        adopted because the suppression is sustained for weeks rather than
        delivered as impulses.
      * `anc_baseline` is a typical adult reference, not this patient's count.
      * `drug_weights`: drug B's marrow potency per mg/m2 is ASSUMED. It is
        swept, never set to 0.0 to make a calendar feasible.

    No neutrophil count in this repository was measured on a patient, and
    nothing here is clinical decision support.
    """

    anc_baseline: float = 4.5          # 10^9/L, typical adult reference
    anc_floor: float = LABEL_ANC_FLOOR
    # Per-drug marrow potency, mg/m2 of drug -> mg/m2 of marrow burden. Drug B
    # is an ASSUMPTION with nothing behind it; see `unknown_drug_potency`.
    drug_weights: Dict[str, float] = field(
        default_factory=lambda: {"A": 1.0, "B": 1.0}
    )
    # Log-normal delay kernel. Fitted to LABEL_QUOTES["nadir_day"]; see the
    # class docstring for what this does and does not determine.
    kernel_log_mu: float = 3.25
    kernel_log_sd: float = 0.16
    kernel_tail_days: int = 460
    # Floating-point slack so the regimen the budget is DEFINED by is not
    # rejected by one unit in the last place.
    tolerance: float = 1e-9

    def unknown_drug_potency(self, schedule: Schedule) -> Tuple[str, ...]:
        """Drugs whose marrow potency in this model is assumed, not measured."""
        return tuple(d for d in schedule.drugs_used() if d != "A")

    # -- the reservoir ---------------------------------------------------
    def _kernel(self) -> np.ndarray:
        t = np.arange(self.kernel_tail_days, dtype=float)
        t[0] = 1e-9
        mu, sd = self.kernel_log_mu, self.kernel_log_sd
        lg = (
            -((np.log(t) - mu) ** 2) / (2.0 * sd * sd)
            - np.log(t)
            - math.log(sd * math.sqrt(2.0 * math.pi))
        )
        w = np.exp(lg)
        w[0] = 0.0            # no instantaneous effect: marrow does not respond
        return w              # to a dose-day in the same instant

    def marrow_exposure(self, schedule: Schedule, horizon_days: int) -> np.ndarray:
        """Marrow-suppression reservoir S(t), one value per day."""
        if horizon_days <= 0:
            return np.zeros(0, dtype=float)
        w = dict(self.drug_weights)
        daily = np.fromiter(
            (schedule.daily_exposure(d, w) for d in range(horizon_days)),
            dtype=float, count=horizon_days,
        )
        return np.convolve(daily, self._kernel())[:horizon_days]

    def peak_suppression(self, schedule: Schedule, horizon_days: int) -> float:
        e = self.marrow_exposure(schedule, horizon_days)
        return float(e.max()) if e.size else 0.0

    def suppression_budget(self) -> float:
        """The reservoir produced by the label's OWN regimen.

        This is the definition of the budget, not a fitted constant: it is
        computed by running this model's own kernel on the regimen quoted in
        the label (150 mg/m2, days 1-5 of a 28-day cycle).
        """
        horizon = LABEL_CYCLE_DAYS * 3
        label_regimen = Schedule.cyclic(
            "A", LABEL_DOSE_MG_M2, LABEL_ON_DAYS, LABEL_CYCLE_DAYS, horizon
        )
        return self.peak_suppression(label_regimen, horizon)

    def max_continuous_mg_m2(self) -> float:
        """Highest CONSTANT marrow-weighted mg/m2/day this screen permits.

        Readable because the kernel has unit area, so a constant dose d leaves
        a steady reservoir of d.
        """
        return self.suppression_budget()

    # -- the count -------------------------------------------------------
    def anc_trajectory(self, schedule: Schedule, horizon_days: int) -> np.ndarray:
        """Modelled ANC, 10^9/L, one value per day (index 0 is day 1)."""
        budget = self.suppression_budget()
        e = self.marrow_exposure(schedule, horizon_days)
        if e.size == 0:
            return np.zeros(0, dtype=float)
        if budget <= 0.0:
            return np.full(e.shape, float(self.anc_baseline))
        frac = np.maximum(
            0.0,
            1.0 - (1.0 - self.anc_floor / self.anc_baseline) * (e / budget),
        )
        return self.anc_baseline * frac

    def nadir(self, schedule: Schedule, horizon_days: int) -> float:
        traj = self.anc_trajectory(schedule, horizon_days)
        return float(traj.min()) if traj.size else float(self.anc_baseline)

    def nadir_day(
        self,
        schedule: Schedule,
        horizon_days: int,
        rel_tol: float = 1e-3,
    ) -> Optional[int]:
        """Day of the modelled nadir, counted 1-based as the label counts.

        A REPEATING regimen's nadirs are equal to within the model's measured
        non-cumulativity -- `anchor_checks` reports that drift, and it is a
        few parts in a hundred thousand of the count. Plain `argmin` would
        therefore answer with whichever later cycle floating-point noise made
        deepest, an answer that moves when the kernel is refitted and means
        nothing. `rel_tol` is set deliberately ABOVE that measured drift, and
        the FIRST day that reaches the minimum is reported.
        """
        traj = self.anc_trajectory(schedule, horizon_days)
        if traj.size == 0:
            return None
        deepest = float(traj.min())
        hit = np.nonzero(traj <= deepest + abs(deepest) * rel_tol)[0]
        return int(hit[0]) + 1 if hit.size else None

    def feasible(self, schedule: Schedule, horizon_days: int) -> bool:
        return self.nadir(schedule, horizon_days) >= self.anc_floor - self.tolerance

    # -- the anchors, measured -------------------------------------------
    def label_regimen(self, horizon_days: int) -> Schedule:
        return Schedule.cyclic(
            "A", LABEL_DOSE_MG_M2, LABEL_ON_DAYS, LABEL_CYCLE_DAYS, horizon_days
        )

    def anchor_checks(self, horizon_days: int = 365) -> Dict:
        """MEASURE the label anchors, instead of asserting that they hold.

        Each key is a property the screen is supposed to have, and each value
        is computed by running the model. A reader does not have to trust this
        file: `tests/test_dosing.py` asserts the same numbers.
        """
        reg = self.label_regimen(horizon_days)
        one_more = Schedule.cyclic(
            "A", LABEL_DOSE_MG_M2, LABEL_ON_DAYS + 1, LABEL_CYCLE_DAYS, horizon_days
        )
        traj = self.anc_trajectory(reg, horizon_days)
        budget = self.suppression_budget()
        first_cycle = traj[: LABEL_CYCLE_DAYS]
        # "generally not cumulative": the deepest point of a later cycle must
        # not sit meaningfully below the first cycle's.
        per_cycle = [
            float(traj[c * LABEL_CYCLE_DAYS:(c + 1) * LABEL_CYCLE_DAYS].min())
            for c in range(horizon_days // LABEL_CYCLE_DAYS)
        ]
        deepest_later = min(per_cycle[1:]) if len(per_cycle) > 1 else per_cycle[0]
        return {
            "label_regimen_feasible": self.feasible(reg, horizon_days),
            "label_regimen_nadir_anc": float(traj.min()),
            "label_regimen_nadir_day_first_cycle": int(first_cycle.argmin()) + 1,
            "label_regimen_nadir_day_1based": self.nadir_day(reg, horizon_days),
            "label_regimen_peak_over_budget": (
                float(self.marrow_exposure(reg, horizon_days).max()) / budget
                if budget > 0 else None
            ),
            "one_extra_day_feasible": self.feasible(one_more, horizon_days),
            "one_extra_day_nadir_anc": self.nadir(one_more, horizon_days),
            "nadir_day_residual_days": (
                int(first_cycle.argmin()) + 1 - LABEL_NADIR_DAY
            ),
            "worst_later_cycle_nadir_ratio": (
                deepest_later / float(first_cycle.min())
                if first_cycle.min() > 0 else None
            ),
            "cycles_non_cumulative_ratio_note": (
                "1.0 means the deepest later-cycle nadir equals the first "
                "cycle's; the label states myelosuppression 'was generally "
                "not cumulative'"
            ),
            "max_continuous_mg_m2_per_day": self.max_continuous_mg_m2(),
            "suppression_budget_mg_m2": budget,
        }

    def to_dict(self) -> Dict:
        return {
            "anc_baseline": self.anc_baseline,
            "anc_floor": self.anc_floor,
            "anc_floor_source": TEMOZOLOMIDE_LABEL_URL,
            "drug_weights": dict(self.drug_weights),
            "kernel": {
                "family": "log-normal marrow-suppression delay kernel",
                "log_mu": self.kernel_log_mu,
                "log_sd": self.kernel_log_sd,
                "tail_days": self.kernel_tail_days,
                "unit_area": True,
                "fitted_to": "the label's reported median day of neutrophil "
                             "nadir, and nothing else",
            },
            "suppression_budget_mg_m2": self.suppression_budget(),
            "budget_definition": "the marrow reservoir produced by the label's "
                                 "own 150 mg/m2 x5d every 28d regimen",
            "nadir_day_rule": "argmin of the modelled ANC + 1, counted 1-based "
                              "as the label counts days",
            "label_quotes": {
                k: {"quote": q, "section": s} for k, (q, s) in LABEL_QUOTES.items()
            },
            "second_drug_potency": "ASSUMED; swept over declared values, never "
                                    "set to zero to admit a calendar",
            "anchors": self.anchor_checks(),
            "status": "anchored_to_fda_label_with_declared_kernel_shape",
            "is_label_threshold": True,
            "measured_in_this_repository": False,
            "is_clinical_decision_support": False,
        }


@dataclass(frozen=True)
class SafetyReport:
    feasible: bool
    anc_nadir: float
    floor: float
    violations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "feasible": self.feasible,
            "anc_nadir": self.anc_nadir,
            "floor": self.floor,
            "violations": list(self.violations),
        }


def screen_schedule(
    schedule: Schedule,
    horizon_days: int,
    model: MyelosuppressionModel = MyelosuppressionModel(),
) -> SafetyReport:
    """Screen a calendar against the declared myelosuppression model."""
    nadir = model.nadir(schedule, horizon_days)
    ok = model.feasible(schedule, horizon_days)
    return SafetyReport(
        feasible=ok,
        anc_nadir=nadir,
        floor=model.anc_floor,
        violations=() if ok else (
            f"declared ANC nadir {nadir:.2f} x10^9/L falls below the "
            f"{model.anc_floor:.2f} floor this module screens against",
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation against the real engine
# ---------------------------------------------------------------------------


def _aubc(traj_burdens: np.ndarray) -> float:
    """Area under the burden curve, normalised by day-0 burden.

    This is the repo's existing outcome measure (see danon.rl_steering), not a
    new one introduced here: lower is better, and a policy that never shrinks
    the tumour scores ~1.
    """
    b0 = float(traj_burdens[0])
    if b0 <= 0:
        return float("nan")
    return float(np.mean(traj_burdens) / b0)


@dataclass
class ScheduleScore:
    name: str
    schedule: Schedule
    seeds: Tuple[int, ...]
    median_aubc: float
    iqr_aubc: Tuple[float, float]
    median_dominance_day: float
    median_pfs_days: float
    median_toxicity: float
    n_progressed: int
    feasibility: SafetyReport
    engine_runs: int
    per_seed: Dict[str, List] = field(default_factory=dict)

    def to_dict(self, with_per_seed: bool = False) -> Dict:
        out = {
            "name": self.name,
            "schedule": self.schedule.to_dict(),
            "seeds": list(self.seeds),
            "n_seeds": len(self.seeds),
            "median_aubc": self.median_aubc,
            "iqr_aubc": list(self.iqr_aubc),
            "median_dominance_day": self.median_dominance_day,
            "median_pfs_days": self.median_pfs_days,
            "median_toxicity": self.median_toxicity,
            "n_progressed": self.n_progressed,
            "feasible": self.feasibility.feasible,
            "anc_nadir": self.feasibility.anc_nadir,
            "engine_runs": self.engine_runs,
        }
        if with_per_seed:
            out["per_seed"] = self.per_seed
        return out


def evaluate_schedule(
    name: str,
    schedule: Schedule,
    sim: TumorEvolutionSimulator,
    seeds: Sequence[int],
    days: int = 365,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
    batch_size: int = MEASURED_BATCH_OPTIMUM,
) -> ScheduleScore:
    """Run ONE calendar on the real engine across `seeds` replicates.

    Each replicate gets its own SchedulePolicy instance because `run_batch`
    resets policies and the same object reused across replicates would carry
    state between them.

    The replicates are chunked at `MEASURED_BATCH_OPTIMUM`, which is 1. Passing
    all of a 30-seed run into one `run_batch` call looks tidier and is about
    four times SLOWER here, for the reason `measure_batch_scaling` documents:
    the batched per-replicate arrays stop fitting in cache and the bandwidth
    cost swamps the call-overhead saving. Chunking is a scheduling change
    only -- each replicate still gets its own `run_batch` and its own rng
    stream -- and `tests/test_dosing.py` pins that the score is identical
    either way, so this is a speed default, never a numerical one.
    """
    specs = [
        TrajectorySpec(
            policy=SchedulePolicy(schedule, name=name),
            seed=int(s),
            initial_resistant_fraction=initial_resistant_fraction,
        )
        for s in seeds
    ]
    if not specs:
        raise ValueError("evaluate_schedule needs at least one seed")
    chunk = max(1, min(int(batch_size), len(specs)))
    trajs: List = []
    for start in range(0, len(specs), chunk):
        trajs.extend(
            sim.run_batch(
                specs[start:start + chunk], days, growth_noise=growth_noise
            )
        )
    burdens = np.array([t.burden_S + t.burden_R for t in trajs], dtype=float)
    aubcs = np.array([_aubc(b) for b in burdens], dtype=float)
    dom = [
        t.resistant_dominance_day if t.resistant_dominance_day is not None else days
        for t in trajs
    ]
    return ScheduleScore(
        name=name,
        schedule=schedule,
        seeds=tuple(int(s) for s in seeds),
        median_aubc=float(np.median(aubcs)),
        iqr_aubc=(float(np.percentile(aubcs, 25)), float(np.percentile(aubcs, 75))),
        median_dominance_day=float(np.median(dom)),
        median_pfs_days=float(np.median([t.pfs_value(days) for t in trajs])),
        median_toxicity=float(np.median([t.toxicity for t in trajs])),
        n_progressed=int(sum(1 for t in trajs if t.progressed)),
        feasibility=screen_schedule(schedule, days, myelo),
        engine_runs=len(trajs),
        per_seed={
            "seed": [int(s) for s in seeds],
            "aubc": [float(a) for a in aubcs],
            "dominance_day": [int(x) for x in dom],
            "pfs_day": [int(t.pfs_value(days)) for t in trajs],
            "progressed": [bool(t.progressed) for t in trajs],
        },
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    horizon_days: int
    seeds: Tuple[int, ...]
    scored: List[ScheduleScore]
    best: Optional[ScheduleScore]
    baselines: Dict[str, ScheduleScore]
    beats_baselines: bool
    engine_runs: int
    candidates_infeasible: int
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "horizon_days": self.horizon_days,
            "n_seeds": len(self.seeds),
            "seeds": list(self.seeds),
            "candidates_evaluated": len(self.scored),
            "candidates_infeasible": self.candidates_infeasible,
            "engine_runs": self.engine_runs,
            "beats_baselines": self.beats_baselines,
            "best": self.best.to_dict() if self.best else None,
            "scored": [s.to_dict() for s in self.scored],
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()},
            "notes": list(self.notes),
        }


def measure_worker_scaling(
    sim: TumorEvolutionSimulator,
    workers: Sequence[int] = (1, 2, 4, 8),
    n_candidates: int = 4,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5),
    days: int = 365,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
) -> List[Dict]:
    """MEASURE how candidate-level parallelism scales, on this machine.

    `evaluate_candidates` can spread whole calendars across processes. Whether
    that is a win is a property of the machine and the grid size, so it is
    measured here rather than assumed: the same rule that made batching
    inside `run_batch` counter-productive applies to any cache- and
    bandwidth-bound workload.

    It also checks the thing that matters more than speed -- that the worker
    count changes NO number. A parallel path that quietly altered a median
    would be worse than a slow one.
    """
    cands = single_drug_candidates(days)[:n_candidates]
    rows: List[Dict] = []
    for w in workers:
        t = time.time()
        scores = evaluate_candidates(
            cands, sim, seeds, days=days, myelo=myelo, max_workers=w,
        )
        wall = time.time() - t
        rows.append({
            "workers": int(w),
            "wall_seconds": round(wall, 2),
            "seconds_per_candidate": round(wall / max(1, len(cands)), 2),
            "median_aubc": [float(s.median_aubc) for s in scores],
        })
    identical = len({tuple(r["median_aubc"]) for r in rows}) == 1
    best = min(rows, key=lambda r: r["seconds_per_candidate"])
    for r in rows:
        r["numbers_identical_across_worker_counts"] = identical
    rows.append({
        "measured_worker_optimum": int(best["workers"]),
        "speedup_vs_one_worker": round(
            rows[0]["seconds_per_candidate"] / best["seconds_per_candidate"], 3
        ) if best["seconds_per_candidate"] else None,
        "n_candidates": len(cands),
        "seeds": len(list(seeds)),
        "days": days,
        "is_speedup_linear_in_workers": (
            best["seconds_per_candidate"] * best["workers"]
            < 0.8 * rows[0]["seconds_per_candidate"]
        ),
    })
    return rows

def measure_batch_scaling(
    sim: TumorEvolutionSimulator,
    sizes: Sequence[int] = (1, 4, 8, 16),
    days: int = 365,
    schedule: Optional[Schedule] = None,
    seed: int = 0,
) -> List[Dict]:
    """MEASURE how per-replicate cost varies with batch size, on this machine.

    `run_batch` exists to amortise numpy call overhead across replicates. That
    pays off only while the batched arrays still fit in cache. Measured here at
    24^3 with float64 fields, per-replicate cost RISES steeply with batch size
    -- a few MB per live field times ~40 fields overruns the cache and the
    bandwidth cost swamps the call-overhead saving. The consequence is that the
    fastest path is often the sequential one, which is the opposite of the
    intuition the batching was built on.

    This function measures it rather than assuming it, so the caller picks a
    batch size from evidence. It is reported in the artifact.
    """
    sched = schedule or Schedule.cyclic("A", 90.0, 5, 28, days)
    rows: List[Dict] = []
    for b in sizes:
        specs = [TrajectorySpec(policy=SchedulePolicy(sched), seed=seed + i) for i in range(b)]
        t0 = time.perf_counter()
        sim.run_batch(specs, days)
        dt = time.perf_counter() - t0
        rows.append({
            "batch_size": int(b),
            "total_seconds": round(dt, 3),
            "ms_per_replicate": round(dt / b * 1000.0, 1),
        })
    return rows


def evaluate_many(
    named_schedules: Sequence[Tuple[str, Schedule]],
    sim: TumorEvolutionSimulator,
    seeds: Sequence[int],
    days: int = 365,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
    batch_size: int = MEASURED_BATCH_OPTIMUM,
) -> List[ScheduleScore]:
    """Evaluate MANY calendars, chunked at a MEASURED-optimal batch size.

    `run_batch` amortises numpy call overhead across replicates, but at 24^3 the
    batched arrays stop fitting in cache and per-replicate cost climbs sharply
    (see `measure_batch_scaling`). The default chunk of 1 is what that
    measurement says is fastest on this machine; it is a measured default, not a
    guess, and callers on other hardware should re-measure.

    This delegates to `evaluate_schedule`, which owns the chunking, so the rule
    exists in exactly one place. At chunk 1 the interleaving it used to do
    itself was the same set of single-replicate `run_batch` calls anyway.

    Chunking is a pure scheduling change: each replicate still gets its OWN
    policy instance and its OWN rng stream in its own `run_batch` call, so a
    candidate's score cannot depend on which chunk it landed in.
    `tests/test_dosing.py::TestBatchedEvaluation` pins that equality.
    """
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        return []
    return [
        evaluate_schedule(
            name, sched, sim, seeds, days, initial_resistant_fraction,
            growth_noise, myelo, batch_size,
        )
        for name, sched in named_schedules
    ]


def feasible_grid_schedules(
    on_days: Iterable[int] = (3, 5, 7, 10),
    cycle_days: Iterable[int] = (28, 42),
    dose_levels: Iterable[float] = (0.4, 0.6, 0.8),
    dose_mg_m2: float = 150.0,
    days: int = 365,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
) -> List[Tuple[str, Schedule]]:
    """The FEASIBLE cells of a declared grid, as (name, Schedule) pairs.

    Feasibility is the declared myelosuppression screen, which costs no engine
    time, so it runs before anything is simulated. The declared grid size and
    the number screened out are both reported by `search_schedules`.

    Shared by the search and by `screen_fidelity` so the two calibrate and rank
    exactly the same candidate set -- if they disagreed, the fidelity number
    would describe a grid nobody searches.
    """
    out: List[Tuple[str, Schedule]] = []
    for on, cyc, level, sched in _grid_schedules(
        on_days, cycle_days, dose_levels, dose_mg_m2, days
    ):
        if not screen_schedule(sched, days, myelo).feasible:
            continue
        out.append((f"{dose_mg_m2 * level:g} mg/m2 x{on}d every {cyc}d", sched))
    return out


def _grid_schedules(
    on_days: Iterable[int],
    cycle_days: Iterable[int],
    dose_levels: Iterable[float],
    dose_mg_m2: float,
    days: int,
) -> List[Tuple[int, int, float, Schedule]]:
    """Enumerate the declared grid. Returns (on, cycle, level, schedule)."""
    out: List[Tuple[int, int, float, Schedule]] = []
    for on in on_days:
        for cyc in cycle_days:
            if on > cyc:
                continue
            for level in dose_levels:
                out.append((
                    on, cyc, level,
                    Schedule.cyclic(
                        "A", dose_mg_m2 * level, on, cyc, days, dose_level=level
                    ),
                ))
    return out


def grid_size(
    on_days: Iterable[int],
    cycle_days: Iterable[int],
    dose_levels: Iterable[float],
) -> int:
    """How many cells the declared grid holds, before any feasibility screen."""
    n = 0
    for on in on_days:
        for cyc in cycle_days:
            if on > cyc:
                continue
            for _ in dose_levels:
                n += 1
    return n


@dataclass
class CombinationComparison:
    """Alternating two-drug calendars vs the best single-drug calendar.

    The comparison is run on the SAME seeds and the SAME engine, so the only
    difference between the arms is the schedule. The verdict is allowed to be
    that alternating does NOT win.

    Drug B's marrow potency is an ASSUMPTION, so the verdict is reported at a
    declared conservative weight AND swept: `marrow_weight_sweep` holds one row
    per assumed weight, and `primary_marrow_weight` is the one the headline
    fields describe. The engine scores do not depend on that weight -- only
    which calendars are permitted does.
    """

    days: int
    seeds: Tuple[int, ...]
    single_drug: ScheduleScore
    marrow_weight_sweep: List[Dict]
    primary_marrow_weight: float
    combination: List[ScheduleScore]
    best_combination: Optional[ScheduleScore]
    beats_single: Optional[bool]
    aubc_ratio: Optional[float]
    dominance_delta_days: Optional[float]
    single_drug_scored: List[ScheduleScore] = field(default_factory=list)
    n_candidates: int = 0
    n_feasible: int = 0
    n_screened_out: int = 0
    engine_runs: int = 0
    parallel_workers: int = 1
    wall_seconds: float = 0.0
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "days": self.days,
            "n_seeds": len(self.seeds),
            "single_drug_best": self.single_drug.to_dict(),
            "single_drug_calendar_count": len(self.single_drug_scored),
            "single_drug_calendars": [s.to_dict() for s in self.single_drug_scored],
            "n_alternating_candidates": self.n_candidates,
            "n_feasible_at_primary_weight": self.n_feasible,
            "n_screened_out_at_every_weight": self.n_screened_out,
            "primary_marrow_weight": self.primary_marrow_weight,
            "primary_weight_is": "the most conservative declared weight: drug B "
                                 "assumed fully myelosuppressive per mg/m2",
            "marrow_weight_sweep": self.marrow_weight_sweep,
            "combination_scored": [s.to_dict() for s in self.combination],
            "best_combination": (
                self.best_combination.to_dict() if self.best_combination else None
            ),
            "beats_single": self.beats_single,
            "aubc_ratio_combination_over_single": self.aubc_ratio,
            "dominance_delta_days": self.dominance_delta_days,
            "engine_runs": self.engine_runs,
            "parallel_workers": self.parallel_workers,
            "wall_seconds": self.wall_seconds,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Combination vs single drug
# ---------------------------------------------------------------------------

# The declared grids. A grid that contained only feasible calendars would
# measure nothing, so each family deliberately spans the safety boundary:
# `ALTERNATION_SHAPES` includes shapes the label-anchored screen rejects, and
# `CONTINUOUS_LEVEL_A` brackets the continuous marrow budget.
DEFAULT_SINGLE_ON_DAYS = (1, 2, 3, 4, 5)
DEFAULT_SINGLE_CYCLE_DAYS = (14, 21, 28, 35, 42)
DEFAULT_SINGLE_CONTINUOUS_LEVELS = (0.2, 0.25, 0.3)
# (block_days, rest_days). (5, 23) is "A five days, off three weeks, B five
# days, off three weeks" -- marrow-equivalent to the label's own regimen, and
# therefore the interesting arm. Shorter rests are included precisely because
# the screen is expected to reject them.
DEFAULT_ALTERNATION_SHAPES = ((5, 23), (5, 25), (5, 16), (5, 9), (3, 25), (3, 9))
DEFAULT_DRUG_B_LEVELS = (0.5, 1.0)
DEFAULT_CONTINUOUS_BLOCKS = (3, 7, 14)
DEFAULT_CONTINUOUS_LEVEL_A = (0.2, 0.3)
# Declared sweep over drug B's marrow potency. 0.0 is deliberately NOT in the
# list: asserting the second drug costs no marrow would manufacture a feasible
# calendar rather than measure one.
DEFAULT_MARROW_WEIGHTS = (1.0, 0.5, 0.3)


def single_drug_candidates(
    days: int,
    dose_mg_m2: float = LABEL_DOSE_MG_M2,
    on_days: Sequence[int] = DEFAULT_SINGLE_ON_DAYS,
    cycle_days: Sequence[int] = DEFAULT_SINGLE_CYCLE_DAYS,
    continuous_levels: Sequence[float] = DEFAULT_SINGLE_CONTINUOUS_LEVELS,
) -> List[Tuple[str, Schedule]]:
    """The reference arm: every single-drug calendar the grid declares.

    This is the BEST single-drug calendar, found by measurement, not a
    hand-picked calendar. Picking one pulsed schedule and calling it "the
    single-drug arm" would quietly decide the comparison in advance.
    """
    out: List[Tuple[str, Schedule]] = []
    for on in on_days:
        for cyc in cycle_days:
            out.append((
                f"single-drug A: {dose_mg_m2:g} mg/m2 x{on}d every {cyc}d",
                Schedule.cyclic("A", dose_mg_m2, on, cyc, days),
            ))
    for lvl in continuous_levels:
        out.append((
            f"single-drug A: {dose_mg_m2:g} mg/m2 continuous at {lvl:g} x full",
            Schedule.continuous("A", dose_mg_m2, days, dose_level=lvl),
        ))
    return out


def alternation_candidates(
    days: int,
    dose_mg_m2: float = LABEL_DOSE_MG_M2,
    shapes: Sequence[Tuple[int, int]] = DEFAULT_ALTERNATION_SHAPES,
    drug_b_levels: Sequence[float] = DEFAULT_DRUG_B_LEVELS,
    continuous_blocks: Sequence[int] = DEFAULT_CONTINUOUS_BLOCKS,
    continuous_level_a: Sequence[float] = DEFAULT_CONTINUOUS_LEVEL_A,
) -> List[Tuple[str, Schedule]]:
    """Every alternating two-drug calendar the grid declares.

    Two families, because "alternate two drugs over time" has two meanings the
    safety model treats completely differently:

      * cycle-scale -- A for `block_days`, off `rest_days`, B for `block_days`,
        off `rest_days`. Both drugs at the clinical face of drug A. The only
        shape the screen permits at full mg/m2 is a long rest.
      * continuous -- A for `block_days`, then B for `block_days`, no rest, at
        reduced engine intensity. Continuous two-drug dosing at 150 mg/m2/day
        exceeds the marrow budget on drug A alone, so this family lives below
        that dose and is declared as doing so.
    """
    out: List[Tuple[str, Schedule]] = []
    for blk, rest in shapes:
        for lb in drug_b_levels:
            out.append((
                f"alternating A/B cycles: A {dose_mg_m2:g} mg/m2 x{blk}d, "
                f"off {rest}d, B {dose_mg_m2:g} mg/m2 at {lb:g} x A x{blk}d, "
                f"off {rest}d",
                Schedule.alternating_relative(
                    lb, blk, days, dose_mg_m2=dose_mg_m2, rest_days=rest,
                ),
            ))
    for blk in continuous_blocks:
        for la in continuous_level_a:
            for lb in drug_b_levels:
                out.append((
                    f"alternating A/B continuous: A {dose_mg_m2:g} mg/m2 at "
                    f"{la:g} x full x{blk}d, B {dose_mg_m2:g} mg/m2 at "
                    f"{lb:g} x full x{blk}d, no rest",
                    Schedule.alternating_relative(
                        lb, blk, days, dose_mg_m2=dose_mg_m2, dose_level_a=la,
                    ),
                ))
    return out


def _evaluate_one(payload):
    """Module-level so a worker process can pickle it."""
    sim, name, schedule, seeds, days, irf, noise, myelo = payload
    return evaluate_schedule(
        name, schedule, sim, seeds, days, irf, noise, myelo
    )


def evaluate_candidates(
    candidates: Sequence[Tuple[str, Schedule]],
    sim: TumorEvolutionSimulator,
    seeds: Sequence[int],
    days: int = 365,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
    max_workers: int = MEASURED_WORKER_OPTIMUM,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[ScheduleScore]:
    """Score many candidates, in parallel ACROSS candidates.

    The parallelism is deliberately placed here and not inside one candidate's
    `run_batch`: `measure_batch_scaling` measured B=1 as optimal on this
    machine, and chunking replicates together is what made the search
    counter-productive. Parallelising over whole candidates leaves that
    measurement intact instead of quietly re-tuning it under a new name.

    The worker count is itself measured (`measure_worker_scaling`): the pool
    saturates well before the core count, so the default is the measured
    optimum rather than `os.cpu_count()`. Results are returned in the order
    given, whatever order the workers finish in.
    """
    cands = list(candidates)
    if not cands:
        return []
    seed_tuple = tuple(int(s) for s in seeds)
    workers = 1 if max_workers <= 1 else min(int(max_workers), len(cands))
    if workers == 1:
        out = []
        for i, (name, sched) in enumerate(cands, 1):
            out.append(evaluate_schedule(
                name, sched, sim, seed_tuple, days,
                initial_resistant_fraction, growth_noise, myelo,
            ))
            if progress:
                progress(i, len(cands), name)
        return out
    payloads = [
        (sim, name, sched, seed_tuple, days, initial_resistant_fraction,
         growth_noise, myelo)
        for name, sched in cands
    ]
    scores: List[Optional[ScheduleScore]] = [None] * len(cands)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_evaluate_one, p): i for i, p in enumerate(payloads)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            scores[i] = fut.result()
            done += 1
            if progress:
                progress(done, len(cands), cands[i][0])
    return [s for s in scores if s is not None]


def compare_combination_vs_single(
    sim: TumorEvolutionSimulator,
    seeds: Sequence[int],
    days: int = 365,
    dose_mg_m2: float = LABEL_DOSE_MG_M2,
    on_days: Sequence[int] = DEFAULT_SINGLE_ON_DAYS,
    cycle_days: Sequence[int] = DEFAULT_SINGLE_CYCLE_DAYS,
    single_continuous_levels: Sequence[float] = DEFAULT_SINGLE_CONTINUOUS_LEVELS,
    alternation_shapes: Sequence[Tuple[int, int]] = DEFAULT_ALTERNATION_SHAPES,
    drug_b_levels: Sequence[float] = DEFAULT_DRUG_B_LEVELS,
    continuous_blocks: Sequence[int] = DEFAULT_CONTINUOUS_BLOCKS,
    continuous_level_a: Sequence[float] = DEFAULT_CONTINUOUS_LEVEL_A,
    marrow_weights: Sequence[float] = DEFAULT_MARROW_WEIGHTS,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
    max_workers: int = MEASURED_WORKER_OPTIMUM,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> CombinationComparison:
    """Measure alternating two-drug calendars against the best single-drug one.

    Both arms are GRIDS, scored on the same seeds on the same engine, so the
    only difference between them is the calendar. The reference arm is the
    best calendar the single-drug grid contains, not a chosen one.

    Every alternating candidate passes the same label-anchored screen as the
    single-drug arm. Drug B's marrow potency is unknown, so the headline
    verdict is taken at the most conservative declared weight (B fully
    myelosuppressive per mg/m2) and the whole verdict is swept over the
    declared weights. The sweep can only change WHICH calendars are permitted,
    never their engine score, because marrow potency is not an engine input.
    """
    started = time.time()
    seeds = tuple(int(s) for s in seeds)
    weights = tuple(sorted({float(w) for w in marrow_weights}, reverse=True))
    if not weights:
        raise ValueError("marrow_weights must not be empty")
    primary_weight = weights[0]

    def with_weight(w: float) -> MyelosuppressionModel:
        return replace(
            myelo, drug_weights={**dict(myelo.drug_weights), "B": w}
        )

    models = {w: with_weight(w) for w in weights}
    single_cands = single_drug_candidates(
        days, dose_mg_m2, on_days, cycle_days, single_continuous_levels,
    )
    combo_cands = alternation_candidates(
        days, dose_mg_m2, alternation_shapes, drug_b_levels,
        continuous_blocks, continuous_level_a,
    )

    notes: List[str] = []
    singles_ok = [
        (n, s) for n, s in single_cands if models[primary_weight].feasible(s, days)
    ]
    if not singles_ok:
        raise ValueError(
            "the declared single-drug grid contains no feasible calendar; the "
            "safety model cannot be compared against anything"
        )
    # An alternating calendar is worth an engine run if it is admissible under
    # ANY declared weight; which weights admit it is decided afterwards, from
    # the same scores, without re-running anything.
    combos_ok = [
        (n, s) for n, s in combo_cands
        if any(models[w].feasible(s, days) for w in weights)
    ]
    n_screened_out = len(combo_cands) - len(combos_ok)
    n_run = len(singles_ok) + len(combos_ok)
    workers = 1 if max_workers <= 1 else min(int(max_workers), n_run)

    scored = evaluate_candidates(
        list(singles_ok) + list(combos_ok), sim, seeds, days,
        initial_resistant_fraction, growth_noise,
        models[primary_weight], workers, progress,
    )
    single_scored = sorted(scored[: len(singles_ok)], key=lambda x: x.median_aubc)
    combo_scored = scored[len(singles_ok):]
    single = single_scored[0]

    sweep: List[Dict] = []
    for w in weights:
        admissible = [
            sc for sc in combo_scored if models[w].feasible(sc.schedule, days)
        ]
        admissible.sort(key=lambda x: x.median_aubc)
        best = admissible[0] if admissible else None
        beats = bool(best and best.median_aubc < single.median_aubc)
        sweep.append({
            "drug_b_marrow_weight": w,
            "n_alternating_calendars_admissible": len(admissible),
            "best_name": best.name if best else None,
            "best_median_aubc": best.median_aubc if best else None,
            "best_median_dominance_day": (
                best.median_dominance_day if best else None
            ),
            "beats_single": beats if best else None,
            "aubc_ratio_over_single": (
                best.median_aubc / single.median_aubc
                if best and single.median_aubc else None
            ),
            "dominance_delta_days": (
                best.median_dominance_day - single.median_dominance_day
                if best else None
            ),
        })

    primary_row = next(
        r for r in sweep if r["drug_b_marrow_weight"] == primary_weight
    )
    primary_combos = sorted(
        (sc for sc in combo_scored
         if models[primary_weight].feasible(sc.schedule, days)),
        key=lambda x: x.median_aubc,
    )
    best = primary_combos[0] if primary_combos else None
    beats = primary_row["beats_single"]
    ratio = primary_row["aubc_ratio_over_single"]
    delta = primary_row["dominance_delta_days"]

    notes.append(
        f"single-drug arm is a grid of {len(single_scored)} calendars; the best "
        f"is {single.name} at median AUBC {single.median_aubc:.4f} "
        f"(dominance day {single.median_dominance_day:.0f}) over {len(seeds)} seeds."
    )
    notes.append(
        f"of {len(combo_cands)} alternating candidates, {n_screened_out} are "
        f"inadmissible under EVERY declared marrow weight and were not run; "
        f"{len(combo_scored)} were run and {primary_row['n_alternating_calendars_admissible']} "
        f"are admissible at the primary weight {primary_weight:g}."
    )
    if best is None:
        notes.append(
            "no alternating calendar is admissible at the primary marrow "
            "weight, so the comparison at that weight is vacuous and is "
            "reported as null rather than dressed up."
        )
    else:
        notes.append(
            f"at the conservative marrow weight {primary_weight:g} the best "
            f"alternating calendar is {best.name} at median AUBC "
            f"{best.median_aubc:.4f} vs single-drug {single.median_aubc:.4f} "
            f"(ratio {ratio:.4f}); "
            f"{'alternating WINS' if beats else 'alternating does NOT win'}."
        )
        if not beats:
            notes.append(
                "reported as a NULL result: on these seeds the alternating arm "
                "did not beat the best single-drug calendar."
            )
    verdicts = {r["beats_single"] for r in sweep if r["beats_single"] is not None}
    if len(verdicts) > 1:
        notes.append(
            "the verdict FLIPS across the declared marrow-weight sweep, so it "
            "is a statement about an assumption about drug B's marrow potency "
            "and not about the engine."
        )
    else:
        notes.append(
            "the verdict does NOT flip anywhere in the declared marrow-weight "
            "sweep, so it does not depend on the assumed potency of drug B."
        )

    return CombinationComparison(
        days=days,
        seeds=seeds,
        single_drug=single,
        marrow_weight_sweep=sweep,
        primary_marrow_weight=primary_weight,
        combination=primary_combos,
        best_combination=best,
        beats_single=beats,
        aubc_ratio=ratio,
        dominance_delta_days=None if delta is None else float(delta),
        single_drug_scored=single_scored,
        n_candidates=len(combo_cands),
        n_feasible=len(primary_combos),
        n_screened_out=n_screened_out,
        engine_runs=sum(s.engine_runs for s in scored),
        parallel_workers=max_workers,
        wall_seconds=round(time.time() - started, 1),
        notes=tuple(notes),
    )


@dataclass
class ScreenFidelity:
    """How much a short-horizon screen distorts the real ranking.

    Produced by `screen_fidelity`, which runs every feasible candidate at BOTH
    horizons so the two rankings can be compared directly. The question is not
    "is the screen fast" -- it is "what does the screen throw away".
    """

    full_days: int
    screen_days: int
    n_candidates: int
    top_k: int
    spearman: float
    kendall_tau: float
    full_winner: str
    full_winner_screen_rank: int
    screen_winner: str
    screen_winner_full_rank: int
    caught: int
    missed: Tuple[str, ...]
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "full_days": self.full_days,
            "screen_days": self.screen_days,
            "n_candidates": self.n_candidates,
            "top_k": self.top_k,
            "spearman_rho": self.spearman,
            "kendall_tau": self.kendall_tau,
            "full_horizon_winner": self.full_winner,
            "full_winner_screen_rank": self.full_winner_screen_rank,
            "screen_pick": self.screen_winner,
            "screen_pick_full_rank": self.screen_winner_full_rank,
            "top_k_caught_by_screen": self.caught,
            "top_k_missed_by_screen": list(self.missed),
            "screen_keeps_full_winner": self.full_winner_screen_rank <= self.top_k,
            "notes": list(self.notes),
        }


def _ranks(values: Sequence[float]) -> List[int]:
    """1-based competition ranks, ties sharing the better rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = i + 1
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    if n < 2:
        return float("nan")
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = n * (n - 1) / 2
    return (conc - disc) / total


def screen_fidelity(
    sim: TumorEvolutionSimulator,
    named_schedules: Sequence[Tuple[str, Schedule]],
    seeds: Sequence[int],
    full_days: int = 365,
    screen_days: int = 120,
    top_k: int = 3,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
    batch_size: int = MEASURED_BATCH_OPTIMUM,
) -> ScreenFidelity:
    """MEASURE what a short-horizon screen mis-ranks.

    Runs every candidate at the full horizon AND at the screen horizon on the
    same seeds, then compares the two rankings. This costs a full search, so it
    is a calibration run, not the production path: run it once to decide
    whether `screen_days` is safe to use for this engine and this grid, and
    report the answer rather than assuming the screen is faithful.
    """
    if not 2 <= screen_days < full_days:
        raise ValueError(f"screen_days must be in [2, {full_days})")
    if not named_schedules:
        raise ValueError("no candidates to compare")

    short = evaluate_many(
        named_schedules, sim, seeds, screen_days,
        initial_resistant_fraction, growth_noise, myelo, batch_size,
    )
    long = evaluate_many(
        named_schedules, sim, seeds, full_days,
        initial_resistant_fraction, growth_noise, myelo, batch_size,
    )

    short_aubc = [s.median_aubc for s in short]
    long_aubc = [s.median_aubc for s in long]
    short_rank = _ranks(short_aubc)
    long_rank = _ranks(long_aubc)

    full_winner_i = min(range(len(long)), key=lambda i: long_aubc[i])
    screen_winner_i = min(range(len(short)), key=lambda i: short_aubc[i])

    full_top = sorted(range(len(long)), key=lambda i: long_aubc[i])[:top_k]
    full_top_names = {long[i].name for i in full_top}
    missed = tuple(
        long[i].name
        for i in full_top
        if short_rank[i] > top_k
    )
    caught = len(full_top) - len(missed)

    notes: List[str] = []
    if short_rank[full_winner_i] > top_k:
        notes.append(
            f"the short screen ranks the full-horizon winner "
            f"({long[full_winner_i].name}) {short_rank[full_winner_i]}"
            f"{'th' if short_rank[full_winner_i] not in (1, 2, 3, 11, 12, 13) else ''}"
            f" of {len(short)} -- it would be DROPPED before confirmation."
        )
    if screen_winner_i != full_winner_i:
        notes.append(
            f"the screen picks {short[screen_winner_i].name}, which is rank "
            f"{long_rank[screen_winner_i]} at the full horizon, not the winner."
        )
    else:
        notes.append("the screen picks the same calendar the full horizon does.")

    return ScreenFidelity(
        full_days=full_days,
        screen_days=screen_days,
        n_candidates=len(named_schedules),
        top_k=top_k,
        spearman=_spearman(short_aubc, long_aubc),
        kendall_tau=_kendall_tau(short_aubc, long_aubc),
        full_winner=long[full_winner_i].name,
        full_winner_screen_rank=short_rank[full_winner_i],
        screen_winner=short[screen_winner_i].name,
        screen_winner_full_rank=long_rank[screen_winner_i],
        caught=caught,
        missed=missed,
        notes=tuple(notes),
    )


def screen_fidelity_note(fidelity: ScreenFidelity) -> str:
    """One sentence for the artifact: what the short screen cost."""
    if not fidelity.missed and fidelity.full_winner_screen_rank <= fidelity.top_k:
        return (
            f"short-horizon screen calibration: at {fidelity.screen_days} days the "
            f"ranking correlates with the {fidelity.full_days}-day ranking at "
            f"Spearman {fidelity.spearman:.3f} over {fidelity.n_candidates} "
            f"candidates; all top-{fidelity.top_k} survivors were retained, so no "
            f"candidate was lost to the short screen."
        )
    return (
        f"short-horizon screen calibration: at {fidelity.screen_days} days the "
        f"ranking correlates with the {fidelity.full_days}-day ranking at "
        f"Spearman {fidelity.spearman:.3f}, but {len(fidelity.missed)} of the "
        f"top-{fidelity.top_k} full-horizon candidates would be DROPPED by the "
        f"screen (missed: {'; '.join(fidelity.missed)}). Short screening is NOT "
        f"safe at this horizon on this grid."
    )


def search_schedules(
    sim: TumorEvolutionSimulator,
    seeds: Sequence[int],
    days: int = 365,
    on_days: Iterable[int] = (2, 3, 5, 7, 10, 14),
    cycle_days: Iterable[int] = (14, 21, 26, 28, 42),
    dose_levels: Iterable[float] = (0.4, 0.6, 0.8, 1.0),
    dose_mg_m2: float = 150.0,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
    screen_seeds: int = 4,
    confirm_top: int = 3,
    screen_days: Optional[int] = None,
) -> SearchResult:
    """Search a declared grid of pulsed calendars against the real engine.

    TWO STAGES, BOTH REAL. A full 365-day replicate costs ~0.9 s here, so
    scoring every cell on the full seed set would take hours. Stage 1 runs the
    whole grid on `screen_seeds` seeds; stage 2 re-runs the `confirm_top`
    survivors on the FULL seed set at the FULL horizon and ranks those. The
    reported winner is the stage-2 number -- stage 1 is a cheap filter, not
    the answer, and the winner is allowed to change when it is confirmed.

    SHORTER-HORIZON SCREENING (`screen_days`). Stage 1 can also run fewer days.
    That is a second, independent saving on top of fewer seeds, and it is the
    only one that changes WHICH quantity is being ranked: a 120-day AUBC is not
    a 365-day AUBC, so the screen is a proxy for the real ranking, not a
    cheaper copy of it. A schedule whose advantage only emerges late can be
    ranked badly by a short screen and dropped before stage 2 ever sees it.

    That risk is not hidden here. `screen_fidelity()` measures it directly --
    how many candidates a short screen mis-ranks, and whether it keeps the
    full-horizon winner -- and `screen_fidelity_note()` states it in the notes
    of any search that uses a short screen. `screen_days=None` (the default)
    disables short screening entirely and ranks stage 1 on the full horizon, so
    the saving is opt-in and its cost is reported when taken.

    The grid is declared here and its size is reported. Nothing is pruned
    silently and nothing is extrapolated: every scored row is a calendar that
    actually ran.

    `beats_baselines` is the honest verdict and it is allowed to be False. A
    search that cannot lose is not a search, it is a foregone conclusion.
    """
    seeds = tuple(int(s) for s in seeds)
    screen = seeds[:screen_seeds]
    if screen_days is not None and not 2 <= screen_days < days:
        raise ValueError(
            f"screen_days must be in [2, {days}) to be a saving; got {screen_days}"
        )
    effective_screen_days = days if screen_days is None else int(screen_days)
    notes: List[str] = [
        f"stage 1 screened {grid_size(on_days, cycle_days, dose_levels)} "
        f"grid cells on {len(screen)} seeds at {effective_screen_days} days; "
        f"stage 2 confirmed the top {confirm_top} on all {len(seeds)} seeds "
        f"at {days} days. The winner is the stage-2 number.",
    ]
    if screen_days is not None:
        notes.append(
            f"stage 1 ranked on a {effective_screen_days}-day AUBC, which is a "
            f"PROXY for the {days}-day ranking, not a cheaper copy of it. "
            f"Candidates the short screen under-rates can be dropped before "
            f"confirmation; measure this with danon.dosing.screen_fidelity "
            f"before trusting the winner from a short screen."
        )

    baselines: Dict[str, ScheduleScore] = {}
    baselines["mtd"] = evaluate_schedule(
        "MTD (continuous)",
        Schedule.continuous("A", dose_mg_m2, days),
        sim, seeds, days, initial_resistant_fraction, growth_noise, myelo,
    )
    best_baseline = baselines["mtd"]

    # --- stage 1: cheap screen over the whole declared grid ----------------
    # Feasibility first (no engine time), then ONE batched call for every
    # surviving cell rather than one call per cell.
    n_cells = grid_size(on_days, cycle_days, dose_levels)
    viable = feasible_grid_schedules(
        on_days, cycle_days, dose_levels, dose_mg_m2, days, myelo
    )
    n_infeasible = n_cells - len(viable)
    stage1 = evaluate_many(
        viable, sim, screen, effective_screen_days,
        initial_resistant_fraction, growth_noise, myelo,
    )
    stage1.sort(key=lambda s: s.median_aubc)

    # --- stage 2: confirm the survivors on the FULL seed set ----------------
    scored: List[ScheduleScore] = evaluate_many(
        [(s.name, s.schedule) for s in stage1[:confirm_top]],
        sim, seeds, days,
        initial_resistant_fraction, growth_noise, myelo,
    )
    scored.sort(key=lambda s: s.median_aubc)

    engine_runs = (
        len(screen) * len(stage1)
        + len(seeds) * len(scored)
        + len(seeds) * len(baselines)
    )
    best = scored[0] if scored else None

    beats = False
    if best is not None:
        beats = best.median_aubc < best_baseline.median_aubc
        if not beats:
            notes.append(
                f"no pulsed calendar beat the best baseline "
                f"({best_baseline.name}, AUBC {best_baseline.median_aubc:.4f}); "
                f"the best pulsed calendar reached {best.median_aubc:.4f}. "
                f"Reported as a null result."
            )
    else:
        notes.append(
            "every pulsed calendar in the declared grid was screened out by the "
            "declared myelosuppression model; nothing was ranked."
        )

    return SearchResult(
        horizon_days=days,
        seeds=seeds,
        scored=sorted(scored, key=lambda s: s.median_aubc),
        best=best,
        baselines=baselines,
        beats_baselines=beats,
        engine_runs=engine_runs,
        candidates_infeasible=n_infeasible,
        notes=tuple(notes),
    )





# ---------------------------------------------------------------------------
# The headline experiment: does competitive release actually hold?
# ---------------------------------------------------------------------------


@dataclass
class CompetitiveReleaseResult:
    """MTD vs the best pulsed calendar, measured on the same seeds.

    The hypothesis under test is the one the project is built around: killing
    the sensitive clone with continuous MTD removes the competition that holds
    the resistant clone back, so MTD should reach resistance dominance EARLIER
    than a pulsed calendar that leaves sensitive cells alive.

    Reported whichever way it comes out.
    """

    days: int
    seeds: Tuple[int, ...]
    mtd: ScheduleScore
    pulsed: ScheduleScore
    dominance_delta_days: Optional[float]
    aubc_ratio: Optional[float]
    hypothesis_supported: Optional[bool]
    per_seed: Dict[str, List] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "days": self.days,
            "n_seeds": len(self.seeds),
            "mtd": self.mtd.to_dict(with_per_seed=True),
            "pulsed": self.pulsed.to_dict(with_per_seed=True),
            "dominance_delta_days": self.dominance_delta_days,
            "aubc_ratio": self.aubc_ratio,
            "hypothesis_supported": self.hypothesis_supported,
            "per_seed": self.per_seed,
            "notes": list(self.notes),
        }


def run_competitive_release_experiment(
    on_days: int = 5,
    cycle_days: int = 26,
    dose_mg_m2: float = 150.0,
    size: int = 24,
    seeds: Sequence[int] = tuple(range(30)),
    days: int = 365,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    myelo: MyelosuppressionModel = MyelosuppressionModel(),
) -> CompetitiveReleaseResult:
    """Measure the project's central hypothesis on held-out seeds.

    Uses a FRESH phantom (seed 12345) that is not the training partition, and
    reports the paired per-seed differences so the effect can be judged rather
    than taken on trust.
    """
    geom = make_gbm_phantom(size=size, seed=12345)
    sim = TumorEvolutionSimulator(geom)

    mtd = evaluate_schedule(
        "MTD (continuous)", Schedule.continuous("A", dose_mg_m2, days),
        sim, seeds, days, initial_resistant_fraction, growth_noise, myelo,
    )
    pulsed_sched = Schedule.cyclic("A", dose_mg_m2, on_days, cycle_days, days)
    pulsed = evaluate_schedule(
        f"pulsed {dose_mg_m2:g} mg/m2 x{on_days}d every {cycle_days}d",
        pulsed_sched, sim, seeds, days, initial_resistant_fraction, growth_noise, myelo,
    )

    # Paired per-seed dominance days come straight off the two arms already
    # run -- no second batch. The delta is a paired difference, not a
    # difference of two independently-noisy medians.
    per_seed = {
        "seed": list(mtd.seeds),
        "mtd_dominance_day": list(mtd.per_seed["dominance_day"]),
        "pulsed_dominance_day": list(pulsed.per_seed["dominance_day"]),
        "mtd_aubc": list(mtd.per_seed["aubc"]),
        "pulsed_aubc": list(pulsed.per_seed["aubc"]),
    }
    deltas = [
        p - m
        for p, m in zip(
            per_seed["pulsed_dominance_day"], per_seed["mtd_dominance_day"]
        )
    ]
    delta = float(np.median(deltas)) if deltas else None
    ratio = (pulsed.median_aubc / mtd.median_aubc) if mtd.median_aubc else None

    notes: List[str] = []
    if ratio is not None and abs(ratio - 1.0) < 1e-3:
        notes.append(
            "pulsed and MTD have indistinguishable median AUBC on these seeds: "
            "the pulsed calendar did not change the outcome measure."
        )
    if delta is not None and delta <= 0:
        notes.append(
            f"pulsed reached resistance dominance no later than MTD "
            f"(median delta {delta:+.0f} d). The competitive-release prediction "
            f"-- that leaving sensitive cells alive DELAYS dominance -- is not "
            f"supported at this dose split."
        )

    return CompetitiveReleaseResult(
        days=days,
        seeds=tuple(int(s) for s in seeds),
        mtd=mtd,
        pulsed=pulsed,
        dominance_delta_days=delta,
        aubc_ratio=ratio,
        hypothesis_supported=(delta > 0) if delta is not None else None,
        per_seed=per_seed,
        notes=tuple(notes),
    )


def _n_wins(scored: Sequence[ScheduleScore], baseline: ScheduleScore) -> int:
    """How many scored calendars beat a baseline on median AUBC."""
    return sum(1 for s in scored if s.median_aubc < baseline.median_aubc)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def engine_identity() -> Dict:
    """The engine's source hash, so every figure here is traceable to bytes.

    A number without an engine identity behind it cannot be checked against
    anything. This records the exact bytes of the four modules holding the
    numeric core, using the repository's own hashing so there is one
    definition of it rather than two.
    """
    from danon.tumor_board import engine_source_hash
    return engine_source_hash()


def provenance() -> Dict:
    return {
        "engine": "measured: every AUBC/PFS/dominance figure is a real "
                  "TumorEvolutionSimulator run in this module; the engine's "
                  "source hash is recorded alongside the results",
        "calendar": "measured: Schedule.dose_on is the engine's own dose series",
        "myelosuppression": "DECLARED ASSUMPTION: real arithmetic on assumed "
                            "parameters; not an NCCN, FDA or label threshold; "
                            "not measured in this repository",
        "schedule_search": "measured on the declared grid, whose size is "
                           "reported; the grid was not pruned or extrapolated",
        "clinical_outcome": "NOT CLAIMED: no schedule here is shown to help a "
                            "patient; the engine is simulation-only",
        "what_would_change_this": [
            "bench-measured kill and growth constants for the drug in question",
            "a real myelosuppression dataset to calibrate the ANC model",
            "paired baseline/recurrence imaging to fix patient geometry",
        ],
    }