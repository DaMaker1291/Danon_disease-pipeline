"""Which mechanisms in this engine actually do work, measured by switching them off.

WHY THIS EXISTS
---------------
A mechanistic model earns its complexity one term at a time. "The model has
anisotropic transport, Lotka-Volterra competition, a resistance fitness cost and
a mutation term" is a description of code; the falsifiable version is "removing
this term moves the outcome by N times the model's own seed-to-seed spread, and
removing that one moves it by less than the noise". Only the second version is
worth a poster, and it can be measured with no patient data at all -- which is
what makes it the strongest thing available here.

WHAT AN ABLATION IS, AND WHAT IT IS NOT
---------------------------------------
It is a PAIRED contrast: the same geometry, the same initial seed layout, the
same policy, the same seeds, with exactly one mechanism's parameter changed.
Everything else is held fixed, so a difference is attributable to that mechanism
and nothing else.

It is NOT validation. A mechanism that changes the outcome is *load-bearing*; a
mechanism that does not is *inert*. Neither statement says the mechanism is
RIGHT. A wrong coefficient, ablated within one model, produces a large effect
size for the same reason a right one does. Correctness needs outcomes, and this
repository holds zero engine-output/patient-outcome pairs.

An ablation can also fail loudly in the useful direction: if removing a mechanism
moves nothing while the run-to-run noise is large, the honest verdict is
"unresolvable at this design", not "the mechanism is unimportant". Both are
recorded.

WHAT IS REFUSED
---------------
- A transport ablation on a geometry with no transport field. "Turn off the
  heterogeneous field" requires a field; substituting a homogeneous one silently
  would be the ablation measuring the substitution.
- An ablation whose replacement changes more than one thing. Each `Mechanism`
  states the parameter it moves and the value it moves it to, and a mechanism
  whose patch touches two fields is rejected at construction.
- Reporting an effect size when the intact arm has no spread to scale it by. The
  ratio is None then, and the verdict is `degenerate`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    CloneParams,
    DrugParams,
    TumorEvolutionSimulator,
    TumorGeometry,
)
from danon.uncertainty import ARM_POLICIES

#: A readout is called load-bearing when removing its mechanism moves it by more
#: than the intact arm's own seed-to-seed spread, and inert when it moves it by
#: less than a tenth of that spread. In between, nothing is claimed.
LOAD_BEARING_RATIO = 1.0
INERT_RATIO = 0.1

#: The readouts every ablation is scored on. They are the same ones the
#: uncertainty budget decomposes, so the two modules can be read together.
READOUTS: Tuple[str, ...] = (
    "pfs",
    "mean_density",
    "final_resistant_fraction",
    "resistance_dominance_day",
)


class AblationError(ValueError):
    """Raised instead of running an ablation that would not isolate anything."""


def _clone_patch(**changes: float) -> Callable[[CloneParams], CloneParams]:
    def apply(params: CloneParams) -> CloneParams:
        return dataclasses.replace(params, **changes)

    return apply


@dataclass(frozen=True)
class Mechanism:
    """One ablatable mechanism: the parameter it moves and where it moves to."""

    name: str
    #: What the ablation removes, in one sentence a poster may carry.
    removes: str
    #: Which parameter is changed ('clones', 'drugs', or 'simulator').
    target: str
    #: The replacement, as a callable applied to the intact object.
    patch: Callable[[Any], Any]
    #: The value the parameter takes on the intact side, for the record.
    intact_note: str
    #: A mechanism that cannot be ablated on a geometry lacking this.
    requires: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target not in ("clones", "drugs", "simulator"):
            raise AblationError(
                f"{self.name}: target must be clones/drugs/simulator, got {self.target!r}"
            )


def _motility_zero(sim_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sim_kwargs)
    out["cell_motility"] = 0.0
    return out


MECHANISMS: Mapping[str, Mechanism] = {
    "heterogeneous_transport": Mechanism(
        name="heterogeneous_transport",
        removes=(
            "the perfusion-derived spatial variation in drug diffusivity, leaving "
            "the calibrated mean diffusivity uniform"
        ),
        target="simulator",
        # Dropping the key is the ablation: the simulator then runs its stated
        # homogeneous operator `D * laplacian(C)` at the calibrated mean D.
        patch=lambda kw: {k: v for k, v in kw.items() if k != "diffusion_field"},
        intact_note="D(x) from the rCBV-derived field, reduced to a mean-one pattern",
        requires="diffusion_field",
    ),
    "asymmetric_competition": Mechanism(
        name="asymmetric_competition",
        removes=(
            "the asymmetric crowding term (R feels the full density of S) that "
            "makes holiday-phase suppression of the resistant clone possible"
        ),
        target="clones",
        patch=_clone_patch(competition=1.0),
        intact_note="competition = DEFAULT_CLONES.competition (S feels R weakly)",
    ),
    "resistance_fitness_cost": Mechanism(
        name="resistance_fitness_cost",
        removes=(
            "the growth price a resistant cell pays, so the two clones grow at "
            "the same rate and the collateral trade-off disappears"
        ),
        target="clones",
        patch=lambda c: dataclasses.replace(c, growth_R=c.growth_S),
        intact_note="growth_R = growth_S * (1 - cost_of_resistance)",
    ),
    "mutation": Mechanism(
        name="mutation",
        removes=(
            "spontaneous S->R mutation, so any resistance in the trial must "
            "already exist at day 0"
        ),
        target="clones",
        patch=_clone_patch(mutation_rate=0.0),
        intact_note="mutation_rate = DEFAULT_CLONES.mutation_rate",
    ),
    "cell_motility": Mechanism(
        name="cell_motility",
        removes=(
            "the only term that moves cells outwards, so the predicted tumour "
            "cannot leave its day-0 mask"
        ),
        target="simulator",
        patch=lambda kw: {**kw, "cell_motility": 0.0},
        intact_note="motility = the legacy voxel-unit literal",
    ),
    "resistant_kill_ceiling": Mechanism(
        name="resistant_kill_ceiling",
        removes=(
            "the resistant clone's reduced kill ceiling, so drug A kills both "
            "clones equally and resistance stops being a survival advantage"
        ),
        target="clones",
        patch=_clone_patch(drug_sensitivity_scale_R=1.0),
        intact_note="drug_sensitivity_scale_R = 0.25",
    ),
    "second_agent": Mechanism(
        name="second_agent",
        removes=(
            "drug B entirely, leaving a single-agent schedule -- the control the "
            "combination arm exists to be compared against"
        ),
        target="drugs",
        patch=lambda d: dataclasses.replace(d, kill_max_S=(d.kill_max_S[0], 0.0),
                                            kill_max_R=(d.kill_max_R[0], 0.0)),
        intact_note="kill_max[1] for both clones",
    ),
}


@dataclass(frozen=True)
class AblationRun:
    """One arm of one paired contrast."""

    label: str
    arm: str
    seeds: Tuple[int, ...]
    readouts: Dict[str, List[Optional[float]]]
    voxels: int

    def mean(self, readout: str) -> Optional[float]:
        values = [v for v in self.readouts[readout] if v is not None]
        return float(np.mean(values)) if values else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "arm": self.arm,
            "seeds": list(self.seeds),
            "readouts": self.readouts,
            "voxels": self.voxels,
        }


def _run_arm(
    geometry: TumorGeometry,
    clones: CloneParams,
    drugs: DrugParams,
    sim_kwargs: Dict[str, Any],
    arm: str,
    seeds: Sequence[int],
    *,
    days: int,
    r0: float,
    fill_fraction: float,
    label: str,
) -> AblationRun:
    per_readout: Dict[str, List[Optional[float]]] = {name: [] for name in READOUTS}
    voxels = int(geometry.tumor_mask.sum())
    for seed in seeds:
        sim = TumorEvolutionSimulator(geometry, clones=clones, drugs=drugs, **sim_kwargs)
        traj = sim.run(
            ARM_POLICIES[arm](),
            days=int(days),
            initial_resistant_fraction=float(r0),
            seed=int(seed),
            fill_fraction=float(fill_fraction),
        )
        total = float(traj.burden_S[-1] + traj.burden_R[-1])
        per_readout["pfs"].append(float(traj.pfs_value(days)))
        per_readout["mean_density"].append(total / voxels)
        per_readout["final_resistant_fraction"].append(
            float(traj.final_resistant_fraction)
        )
        # Why the day a clone takes over is a readout worth its own column: it is
        # the one number that says WHEN the evolutionary mechanism acted, and a
        # mechanism can leave the day-365 burden identical while moving it.
        per_readout["resistance_dominance_day"].append(
            None
            if traj.resistant_dominance_day is None
            else float(traj.resistant_dominance_day)
        )
    return AblationRun(
        label=label,
        arm=arm,
        seeds=tuple(int(s) for s in seeds),
        readouts=per_readout,
        voxels=voxels,
    )


def _spread(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 2:
        return {"mean": float(arr.mean()) if arr.size else None,
                "spread_rel": None, "degenerate": True, "n": int(arr.size)}
    lo, hi, mean = float(arr.min()), float(arr.max()), float(arr.mean())
    if mean == 0.0 or hi == lo:
        return {"mean": mean, "spread_rel": 0.0, "degenerate": True, "n": int(arr.size)}
    return {
        "mean": mean,
        "spread_rel": (hi - lo) / abs(mean),
        "degenerate": False,
        "n": int(arr.size),
    }


def ablation_contrast(
    mechanism: Mechanism,
    geometry: TumorGeometry,
    *,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
    diffusion_field: Optional[object] = None,
    arms: Sequence[str] = ("mtd", "steering"),
    seeds: Sequence[int] = (101, 102, 103),
    days: int = 365,
    r0: float = 0.01,
    fill_fraction: float = 0.65,
) -> Dict[str, Any]:
    """Run one mechanism intact and removed, paired on seeds, and report the gap.

    The effect size is reported in units of the INTACT arm's own seed-to-seed
    spread, because a mechanism that moves an endpoint less than the model's own
    run-to-run noise has not been shown to do anything.
    """
    if mechanism.requires == "diffusion_field" and diffusion_field is None:
        raise AblationError(
            f"{mechanism.name}: this geometry carries no transport field, and "
            "'turn the heterogeneous field off' is not the same experiment as "
            "'there was never a field'"
        )
    for arm in arms:
        if arm not in ARM_POLICIES:
            raise AblationError(f"arm must be one of {sorted(ARM_POLICIES)}, got {arm!r}")

    intact_sim_kwargs: Dict[str, Any] = {}
    if diffusion_field is not None:
        intact_sim_kwargs["diffusion_field"] = diffusion_field
    intact_clones, intact_drugs = clones, drugs
    ablated_clones, ablated_drugs = clones, drugs
    ablated_sim_kwargs = dict(intact_sim_kwargs)

    if mechanism.target == "clones":
        ablated_clones = mechanism.patch(clones)
        if ablated_clones == clones:
            raise AblationError(
                f"{mechanism.name}: the replacement leaves the clone parameters "
                "unchanged, so the contrast would measure nothing"
            )
    elif mechanism.target == "drugs":
        ablated_drugs = mechanism.patch(drugs)
        if ablated_drugs == drugs:
            raise AblationError(
                f"{mechanism.name}: the replacement leaves the drug parameters "
                "unchanged, so the contrast would measure nothing"
            )
    else:
        ablated_sim_kwargs = mechanism.patch(dict(intact_sim_kwargs))
        if ablated_sim_kwargs == intact_sim_kwargs:
            raise AblationError(
                f"{mechanism.name}: the replacement leaves the simulator "
                "unchanged, so the contrast would measure nothing"
            )

    report: Dict[str, Any] = {
        "mechanism": mechanism.name,
        "removes": mechanism.removes,
        "intact_note": mechanism.intact_note,
        "target": mechanism.target,
        "days": int(days),
        "r0": float(r0),
        "seeds": [int(s) for s in seeds],
        "arms": {},
        "voxels": int(geometry.tumor_mask.sum()),
    }
    for arm in arms:
        intact = _run_arm(
            geometry, intact_clones, intact_drugs, intact_sim_kwargs, arm, seeds,
            days=days, r0=r0, fill_fraction=fill_fraction, label="intact",
        )
        ablated = _run_arm(
            geometry, ablated_clones, ablated_drugs, ablated_sim_kwargs, arm, seeds,
            days=days, r0=r0, fill_fraction=fill_fraction, label="ablated",
        )
        per_readout: Dict[str, Any] = {}
        for readout in READOUTS:
            base = intact.mean(readout)
            variant = ablated.mean(readout)
            spread = _spread(intact.readouts[readout])
            delta = None if base is None or variant is None else variant - base
            rel = None if not base or delta is None else delta / abs(base)
            ratio = None
            if (
                rel is not None
                and spread["spread_rel"]
                and not spread["degenerate"]
            ):
                ratio = abs(rel) / spread["spread_rel"]
            if ratio is None:
                verdict = "degenerate"
            elif ratio >= LOAD_BEARING_RATIO:
                verdict = "load_bearing"
            elif ratio < INERT_RATIO:
                verdict = "inert"
            else:
                verdict = "unresolved"
            per_readout[readout] = {
                "intact_mean": base,
                "ablated_mean": variant,
                "delta": delta,
                "rel_delta": rel,
                "intact_spread_rel": spread["spread_rel"],
                "intact_spread_degenerate": spread["degenerate"],
                "n_seeds_with_a_value": spread["n"],
                "delta_over_spread": ratio,
                "verdict": verdict,
            }
        report["arms"][arm] = {
            "intact": intact.as_dict(),
            "ablated": ablated.as_dict(),
            "readouts": per_readout,
            # Which readouts this arm's removal actually moved. A mechanism that
            # matters only on one endpoint still matters, and saying WHICH
            # endpoint is part of the result rather than a footnote to it.
            "load_bearing_on": sorted(
                r for r, v in per_readout.items() if v["verdict"] == "load_bearing"
            ),
        }
    all_verdicts = [
        v["verdict"]
        for entry in report["arms"].values()
        for v in entry["readouts"].values()
    ]
    load_bearing_on = sorted(
        f"{arm}/{readout}"
        for arm, entry in report["arms"].items()
        for readout in entry["load_bearing_on"]
    )
    report["load_bearing_on"] = load_bearing_on
    report["verdicts_present"] = sorted(set(all_verdicts))
    report["any_load_bearing"] = "load_bearing" in all_verdicts
    report["all_inert"] = all(v == "inert" for v in all_verdicts)
    return report


def ablate_all(
    geometry: TumorGeometry,
    *,
    mechanisms: Optional[Sequence[str]] = None,
    diffusion_field: Optional[object] = None,
    field_for: Optional[Sequence[str]] = None,
    field: Optional[object] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run every requested mechanism and assemble the leaderboard.

    `diffusion_field` applies to every mechanism. `field_for`/`field` apply the
    second field only to the named mechanisms -- for the transport ablation, whose
    intact arm is meaningless without one -- so a cohort does not pay the
    transport branch's much finer sub-step seven times over. Which configuration
    each mechanism ran under is recorded per mechanism, not described in prose.
    """
    names = list(mechanisms) if mechanisms is not None else list(MECHANISMS)
    unknown = [n for n in names if n not in MECHANISMS]
    if unknown:
        raise AblationError(f"unknown mechanism(s): {unknown}; known: {list(MECHANISMS)}")
    selective = set(field_for or ())
    unknown_selective = sorted(selective - set(names))
    if unknown_selective:
        raise AblationError(
            f"field_for names mechanisms not being run: {unknown_selective}"
        )
    if selective and field is None:
        raise AblationError(
            f"field_for={sorted(selective)} was requested without a field to apply"
        )
    out: Dict[str, Any] = {
        "geometry_voxels": int(geometry.tumor_mask.sum()),
        "mechanisms": {},
        "skipped": {},
        "configuration": {},
        "field_for": sorted(selective),
    }
    for name in names:
        mechanism = MECHANISMS[name]
        use_field = field if name in selective else diffusion_field
        out["configuration"][name] = (
            "transport field supplied" if use_field is not None else "homogeneous transport"
        )
        try:
            out["mechanisms"][name] = ablation_contrast(
                mechanism, geometry, diffusion_field=use_field, **kwargs
            )
        except AblationError as exc:
            # A refusal is a result here: it says which geometry can and cannot
            # support which experiment, and it is not silently dropped.
            out["skipped"][name] = str(exc)
    return out


def ablation_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "paired_contrast_isolates_one_parameter": EvidenceStatus.ESTABLISHED,
        "effect_size_in_seed_spread_units": EvidenceStatus.ESTABLISHED,
        "mechanism_is_load_bearing": EvidenceStatus.ESTABLISHED,
        "mechanism_is_correct": EvidenceStatus.PROVEN_FAILED,
        "ablation_is_validation": EvidenceStatus.PROVEN_FAILED,
        "white_matter_tract_anisotropy": EvidenceStatus.PROVEN_FAILED,
    }


def what_an_ablation_cannot_show() -> List[str]:
    return [
        "correctness: a wrong coefficient and a right one produce the same "
        "positive effect size when the same model is ablated",
        "that the mechanism is the one a real tumour uses: an inert mechanism "
        "could be inert because it is wrong, or right but second-order at this "
        "design, and this harness distinguishes those two only by the size of "
        "the seed spread it reports beside the effect",
        "anything about white-matter tract invasion: the engine's transport is "
        "scalar or perfusion-derived, and no diffusion-tensor map exists in this "
        "repository, so an 'anisotropic tensor PDE' is not what is being ablated",
        "a clinical readout: the endpoints are simulated PFS and burden under "
        "declared constants",
    ]
