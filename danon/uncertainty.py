"""How much of a prediction's spread is the patient, and how much is the model?

WHY THIS MODULE EXISTS
----------------------
Pillar 1 of the project's own statement of purpose asks for "explicit Bayesian
uncertainty bounds". This repository cannot produce a Bayesian credible
interval, and saying so plainly is the first thing this module does:

  * a credible interval is a quantile of a POSTERIOR -- a parameter
    distribution conditioned on a likelihood;
  * a likelihood here would have to relate an engine output to a real patient
    outcome, and the repository holds **zero** such pairs (see
    `what_these_claims_cannot_establish`);
  * so what exists is the *empirical* distribution of a measured input over a
    cohort of donors. Propagating it through the engine gives an interval whose
    coverage is over (donor, seed) jointly. That is a useful interval. It is
    not a posterior, and the field is named `is_credible_interval = False` so
    no renderer can quietly call it one.

WHAT THIS MODULE CAN ANSWER, AND WHY THAT QUESTION MATTERS
----------------------------------------------------------
One question is answerable without a likelihood, and it is the question that
decides whether "patient-specific digital twin" means anything:

    **Is the patient-to-patient spread of the measured input resolvable above
    the model's own seed-to-seed stochasticity, for a given arm and readout?**

If the model's own randomness dominates the measured input, then no amount of
better input data makes the twin patient-specific -- the noise floor is inside
the model. If the input dominates, the input is doing work. Either answer is a
result; the algebra is a one-way random-effects decomposition, stated in full
below rather than hidden behind a library call.

THE DECOMPOSITION, IN FULL
--------------------------
For k inputs (one per donor) with r replicates each, outcome y_ij:

    SSB = r * sum_i (mean_i - grand)^2          MSB = SSB / (k - 1)
    SSW = sum_ij (y_ij - mean_i)^2              MSW = SSW / (k * (r - 1))

    sigma^2_within  = MSW                        <- the model's own stochasticity
    sigma^2_between = (MSB - MSW) / r            <- what the input moves

`sigma^2_between` is an unbiased estimator and can therefore come out NEGATIVE
when the input effect is weaker than the noise. It is NOT clipped silently:
the raw mean squares and the sign are reported, and `input_fraction` is None in
that case because the honest reading is "not resolvable", not "zero".

The degenerate case is handled the same way: an arm whose readout is identical
across every donor and seed has no spread to decompose, so it returns no
fraction rather than 100% or 0%.

WHAT THE DECOMPOSITION IS NOT
-----------------------------
- MGMT promoter beta is ONE scalar input out of the engine's many. It is the
  only input in this repository with a real per-donor distribution (n=140 from
  two probes); the rest are smaller or proxies. The budget below bounds the
  input channel this repository can actually measure, not all of them.
- The mapping beta -> r0 is a DECLARED convention, not a fit, so the band
  inherits that convention. A different anchor moves every number in it.
- The input distribution is a cohort of donor tumours, not the patient being
  simulated. One beta per case is exactly the limit `what_mgmt_cannot_show`
  already states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    CollateralSwitchPolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
    TumorGeometry,
)

#: The two policies a budget is computed for. Names are the config keys the
#: sample runner uses, so an arm means the same thing in both places.
ARM_POLICIES = {"mtd": MTDPolicy, "steering": CollateralSwitchPolicy}

#: Readouts, each a function of a finished trajectory. `pfs_days` saturates at
#: the horizon, which is why the burden readout exists: an arm that never
#: progresses has no spread in PFS at all and would decompose to nothing.
READOUTS = ("pfs_days", "final_burden")

#: The empirical interval reported for a propagated outcome.
BAND_QUANTILES = (2.5, 97.5)


class UncertaintyError(ValueError):
    """A decomposition that cannot be computed, refused by name and reason."""


# ---------------------------------------------------------------------------
# The band
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """An interval over a set of REALISED outcomes, plus the median."""

    lo: float
    median: float
    hi: float
    n: int

    @property
    def width(self) -> float:
        return float(self.hi - self.lo)

    def __str__(self) -> str:
        return f"[{self.lo:.6g}, {self.hi:.6g}] median {self.median:.6g} (n={self.n})"


def percentile_band(values: Sequence[float]) -> Band:
    """The 2.5/97.5 interval of a sample. Refuses a sample of one.

    A one-element band is the value itself with zero width, which reads as
    certainty. It is not a measurement of spread, so it is refused.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        raise UncertaintyError("an empty sample is not a band")
    if arr.size < 2:
        raise UncertaintyError(
            f"a band needs at least 2 outcomes, got {arr.size}: a one-element band "
            "is the value itself and reads as certainty"
        )
    lo, hi = BAND_QUANTILES
    return Band(
        lo=float(np.percentile(arr, lo)),
        median=float(np.median(arr)),
        hi=float(np.percentile(arr, hi)),
        n=int(arr.size),
    )


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceBudget:
    """One-way random-effects decomposition of an outcome over inputs x seeds."""

    n_inputs: int
    n_replicates: int
    df_between: int
    df_within: int
    ms_between: float
    ms_within: float

    @property
    def f_ratio(self) -> Optional[float]:
        """MSB / MSW -- the standard, always-defined reading of 'resolvable?'."""
        if self.ms_within <= 0.0:
            return None
        return float(self.ms_between / self.ms_within)

    @property
    def component_within(self) -> float:
        """sigma^2_within: the model's own stochasticity."""
        return float(self.ms_within)

    @property
    def component_between(self) -> Optional[float]:
        """(MSB - MSW) / r, unbiased. None when it is negative (not resolvable).

        Returning None rather than a clipped zero is the whole point: the
        estimator's sign carries the verdict.
        """
        if self.n_replicates < 1:
            return None
        value = (self.ms_between - self.ms_within) / self.n_replicates
        return float(value) if value >= 0.0 else None

    @property
    def input_fraction(self) -> Optional[float]:
        """Share of total variance the input accounts for, or None if unflagged."""
        between, within = self.component_between, self.component_within
        if between is None:
            return None
        total = between + within
        if total <= 0.0:
            return None
        return float(between / total)

    @property
    def is_degenerate(self) -> bool:
        """No spread of any kind: nothing here can be attributed."""
        return self.ms_between <= 0.0 and self.ms_within <= 0.0

    # The components in the units a reader actually reasons in. The ratio below
    # is the sentence worth quoting: how much further a REPEAT run of the same
    # patient moves the answer than a different patient's input does.

    @property
    def between_sd(self) -> Optional[float]:
        """SD of what the input moves, in the readout's own units."""
        between = self.component_between
        return None if between is None else float(np.sqrt(between))

    @property
    def within_sd(self) -> float:
        """SD of a repeat run of the SAME input, in the readout's own units."""
        return float(np.sqrt(self.component_within))

    @property
    def noise_to_signal_sd_ratio(self) -> Optional[float]:
        """within_sd / between_sd -- >1 means the model is its own limiting factor."""
        between = self.between_sd
        if between is None or between <= 0.0:
            return None
        return float(self.within_sd / between)


def f_p_value(f: float, df_between: int, df_within: int) -> Optional[float]:
    """Upper-tail probability of an F with these df, or None if scipy is absent.

    Scipy is a declared dependency, so this is normally available; the None is
    so a caller in a stripped environment reports 'not computed' rather than a
    guess. An F ratio alone is not a verdict -- df decides what counts as large.
    """
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None
    if df_between <= 0 or df_within <= 0 or f < 0.0:
        return None
    return float(1.0 - stats.f.cdf(float(f), int(df_between), int(df_within)))


def variance_budget(groups: Sequence[Sequence[float]]) -> VarianceBudget:
    """Decompose replicated outcomes into input-driven and stochastic parts.

    `groups[i]` holds the replicate outcomes for input i. Every group must hold
    the SAME number of replicates: with unequal group sizes the between-group
    mean square is no longer an unbiased estimator of the same quantity the
    within-group one is, and mixing them silently would be wrong.
    """
    if len(groups) == 0:
        raise UncertaintyError("no inputs: nothing to decompose")
    sizes = {len(g) for g in groups}
    if len(sizes) != 1:
        raise UncertaintyError(
            f"groups must all hold the same number of replicates, got sizes "
            f"{sorted(sizes)}: unequal sizes make MSB and MSW estimate different "
            "quantities"
        )
    r = sizes.pop()
    if r < 2:
        raise UncertaintyError(
            f"a decomposition needs at least 2 replicates per input, got {r}: with "
            "one replicate the stochastic component is unmeasurable and any "
            "fraction would be MSB with no noise floor to compare it against"
        )
    if len(groups) < 2:
        raise UncertaintyError(
            "a decomposition needs at least 2 inputs: with one input there is no "
            "between-group variance to estimate"
        )

    k = len(groups)
    arr = np.asarray(groups, dtype=float)
    means = arr.mean(axis=1)
    grand = float(arr.mean())
    ss_between = float(r * np.sum((means - grand) ** 2))
    ss_within = float(np.sum((arr - means[:, None]) ** 2))
    return VarianceBudget(
        n_inputs=k,
        n_replicates=r,
        df_between=k - 1,
        df_within=k * (r - 1),
        ms_between=ss_between / (k - 1),
        ms_within=ss_within / (k * (r - 1)),
    )


# ---------------------------------------------------------------------------
# The measurement: a real input distribution through the real engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmBudget:
    """One arm x readout: the propagated band, the input span, the budget."""

    arm: str
    readout: str
    #: The interval induced by drawing one donor's input and one seed -- i.e.
    #: what the model actually answers with across the cohort.
    propagated_band: Band
    #: The interval over per-input medians: what the input alone moves, with the
    #: replicates already averaged out.
    input_span: Band
    budget: VarianceBudget

    def verdict(self) -> str:
        budget = self.budget
        if budget.is_degenerate:
            return (
                f"{self.arm}/{self.readout}: every donor and seed returned the same "
                "value, so there is no spread to attribute"
            )
        between = budget.component_between
        if between is None:
            return (
                f"{self.arm}/{self.readout}: the input effect is NOT resolvable above "
                f"the model's own noise (MSB {budget.ms_between:.6g} < MSW "
                f"{budget.ms_within:.6g}, sigma^2_between = "
                f"{(budget.ms_between - budget.ms_within) / budget.n_replicates:.6g} "
                "<= 0) -- the estimator's sign is the verdict, and it is not clipped"
            )
        fraction = budget.input_fraction
        return (
            f"{self.arm}/{self.readout}: the input accounts for "
            f"{fraction * 100:.1f}% of the total variance (sigma^2_between "
            f"{between:.6g} vs sigma^2_within {budget.component_within:.6g}; "
            f"F={budget.f_ratio:.3g} on {budget.df_between}/{budget.df_within} df)"
            if fraction is not None and budget.f_ratio is not None
            else f"{self.arm}/{self.readout}: degenerate"
        )


@dataclass(frozen=True)
class UncertaintyReport:
    """One geometry, one input distribution, every arm and readout."""

    geometry: str
    n_voxels: int
    n_tumour_voxels: int
    days: int
    initial_resistant_fraction: float
    input_quantity: str
    input_source: str
    inputs: Tuple[float, ...]
    input_band: Band
    arms: Tuple[ArmBudget, ...]
    #: Set by construction, and rendered: this is a propagation of a measured
    #: input distribution, NOT a posterior quantile.
    is_credible_interval: bool = False

    def by(self, arm: str, readout: str) -> ArmBudget:
        for row in self.arms:
            if row.arm == arm and row.readout == readout:
                return row
        raise KeyError(f"no budget for {arm}/{readout}")


def mgmt_input_distribution(panel: Any) -> Tuple[np.ndarray, str]:
    """The per-donor mean MGMT beta over the panel's probes, in a fixed order.

    Per-donor mean over the probes present, missing probes NOT imputed -- the
    panel's own accessor, so this cannot drift from the rest of the repository.
    """
    ids = panel.patient_ids
    values = [panel.mean_beta(pid) for pid in ids]
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 2:
        raise UncertaintyError(
            f"the panel yields {arr.size} usable donor beta(s); a distribution needs "
            "at least 2"
        )
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        raise UncertaintyError(
            f"beta values must lie in [0, 1], got [{arr.min()}, {arr.max()}]"
        )
    return arr, str(panel)


def run_input_grid(
    geometry: TumorGeometry,
    betas: Sequence[float],
    *,
    arm: str,
    days: int,
    replicates: int,
    initial_resistant_fraction: float,
    seed0: int = 0,
    clones_for_beta: Optional[Any] = None,
) -> Dict[str, List[List[float]]]:
    """Run every (input, seed) cell once; return per-readout groups.

    Returns {readout: [[y for each replicate] for each input]}. One
    simulator per input, reused across replicates, so the clone parameters are
    a property of the input and not of the replicate.
    """
    if arm not in ARM_POLICIES:
        raise UncertaintyError(f"arm must be one of {sorted(ARM_POLICIES)}, got {arm!r}")
    if replicates < 2:
        raise UncertaintyError(
            f"replicates must be at least 2 to measure the stochastic component, "
            f"got {replicates}"
        )
    if len(betas) < 2:
        raise UncertaintyError("at least 2 inputs are required")

    if clones_for_beta is None:
        from danon.mgmt import clone_params_for_mgmt_beta

        clones_for_beta = clone_params_for_mgmt_beta

    groups: Dict[str, List[List[float]]] = {name: [] for name in READOUTS}
    for beta in betas:
        sim = TumorEvolutionSimulator(geometry, clones=clones_for_beta(float(beta)))
        per_readout: Dict[str, List[float]] = {name: [] for name in READOUTS}
        for rep in range(replicates):
            traj = sim.run(
                ARM_POLICIES[arm](),
                days=days,
                initial_resistant_fraction=initial_resistant_fraction,
                seed=seed0 + rep,
            )
            per_readout["pfs_days"].append(float(traj.pfs_value(days)))
            per_readout["final_burden"].append(
                float(traj.burden_S[-1] + traj.burden_R[-1])
            )
        for name in READOUTS:
            groups[name].append(per_readout[name])
    return groups


def budget_from_groups(arm: str, groups: Dict[str, List[List[float]]]) -> List[ArmBudget]:
    """Turn a run grid into one ArmBudget per readout."""
    out: List[ArmBudget] = []
    for name in READOUTS:
        per_input = groups[name]
        flat = [v for group in per_input for v in group]
        medians = [float(np.median(group)) for group in per_input]
        out.append(
            ArmBudget(
                arm=arm,
                readout=name,
                propagated_band=percentile_band(flat),
                input_span=percentile_band(medians),
                budget=variance_budget(per_input),
            )
        )
    return out


def propagate_mgmt_uncertainty(
    geometry: TumorGeometry,
    panel: Any,
    *,
    geometry_label: str,
    arms: Sequence[str] = ("mtd", "steering"),
    days: int = 365,
    replicates: int = 3,
    initial_resistant_fraction: float = 0.01,
    seed0: int = 0,
) -> UncertaintyReport:
    """The full report: the measured input distribution through each arm.

    Nothing here is a posterior. The input is the panel's own per-donor mean
    beta, the mapping to r0 is the repository's declared convention, and the
    band's coverage is over (donor, seed) jointly.
    """
    betas, source = mgmt_input_distribution(panel)
    budgets: List[ArmBudget] = []
    for arm in arms:
        budgets.extend(
            budget_from_groups(
                arm,
                run_input_grid(
                    geometry,
                    betas,
                    arm=arm,
                    days=days,
                    replicates=replicates,
                    initial_resistant_fraction=initial_resistant_fraction,
                    seed0=seed0,
                ),
            )
        )
    return UncertaintyReport(
        geometry=geometry_label,
        n_voxels=int(geometry.tumor_mask.size),
        n_tumour_voxels=int(geometry.tumor_mask.sum()),
        days=days,
        initial_resistant_fraction=float(initial_resistant_fraction),
        input_quantity="MGMT promoter methylation beta -> r0 = 0.75 * (1 - beta)",
        input_source=source,
        inputs=tuple(float(b) for b in betas),
        input_band=percentile_band(betas),
        arms=tuple(budgets),
    )


# ---------------------------------------------------------------------------
# Rendering and the honesty envelope
# ---------------------------------------------------------------------------


def render_uncertainty_report(report: UncertaintyReport) -> str:
    """The text block for the logbook. Every line is a measured field."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("VARIANCE BUDGET: WHAT THE PATIENT MOVES vs WHAT THE MODEL MOVES")
    lines.append("=" * 78)
    lines.append(
        f"  geometry: {report.geometry} -- {report.n_tumour_voxels:,} tumour voxels "
        f"of {report.n_voxels:,}"
    )
    lines.append(
        f"  input: {report.input_quantity}  ({len(report.inputs)} donors, band "
        f"{report.input_band})"
    )
    lines.append(f"  trial: {report.days} days, r0(input) fixed per donor")
    lines.append("")
    lines.append(f"  credible interval: {report.is_credible_interval}  (see the module docstring)")
    lines.append("")
    for row in report.arms:
        lines.append(f"  [{row.arm}] {row.readout}")
        lines.append(f"    propagated 95% band (donor x seed): {row.propagated_band}")
        lines.append(f"    input span (per-donor medians)    : {row.input_span}")
        lines.append(f"    {row.verdict()}")
    lines.append("")
    lines.append("What this cannot show:")
    for item in what_this_decomposition_cannot_show():
        lines.append(f"  - {item}")
    return "\n".join(lines)


def uncertainty_provenance() -> Dict[str, EvidenceStatus]:
    """Every quantity here in the repository's vocabulary."""
    return {
        "beta_values": EvidenceStatus.ESTABLISHED,       # read from published probes
        "input_distribution": EvidenceStatus.ESTABLISHED,  # the panel's own betas
        "r0_mapping": EvidenceStatus.UNKNOWN,            # a declared convention
        "variance_decomposition": EvidenceStatus.ESTABLISHED,  # computed, not asserted
        "credible_interval": EvidenceStatus.PROVEN_FAILED,     # no likelihood, no posterior
        "engine_predicts_patients": EvidenceStatus.UNKNOWN,
    }


def what_this_decomposition_cannot_show() -> List[str]:
    """Stated so no number here is read as a posterior or a clinical bound."""
    return [
        "that this is a Bayesian credible interval: a credible interval is a "
        "posterior quantile, a posterior needs a likelihood, and this repository "
        "holds ZERO engine-output/patient-outcome pairs to build one from. The "
        "interval here is the empirical spread of a measured input propagated "
        "through the engine, with coverage over (donor, seed) jointly",
        "that MGMT beta is the only uncertainty in the prediction: it is the one "
        "input this repository can measure per donor, and the engine's rate "
        "constants remain declared assumptions, so the input channel measured "
        "here is a lower bound on the true input uncertainty",
        "that a band from donor tumours bounds the error for a NEW patient: the "
        "one-beta-per-case contract is a cohort distribution applied to a case",
        "that the mapping beta -> r0 is right: it is a declared convention "
        "anchored on the calibrated default, and a different anchor moves every "
        "number in the band without any code changing",
        "that the trial design is fixed: the switching arm saturates at the "
        "horizon, so its PFS readout has no spread to decompose and the "
        "conclusion drawn from it is about THIS design, not the policy family",
    ]


__all__ = [
    "ARM_POLICIES",
    "BAND_QUANTILES",
    "READOUTS",
    "ArmBudget",
    "Band",
    "UncertaintyError",
    "UncertaintyReport",
    "VarianceBudget",
    "budget_from_groups",
    "f_p_value",
    "mgmt_input_distribution",
    "percentile_band",
    "propagate_mgmt_uncertainty",
    "render_uncertainty_report",
    "run_input_grid",
    "uncertainty_provenance",
    "variance_budget",
    "what_this_decomposition_cannot_show",
]
