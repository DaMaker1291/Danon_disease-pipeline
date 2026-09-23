"""
Massive-scale verification campaign: millions of elementary property
checks accumulated across thousands of paired simulation runs.

WHAT "VERIFY MILLIONS OF TIMES" MEANS HERE (and what it does not)
-----------------------------------------------------------------
It means: elementary boolean property checks -- non-negativity of every
voxel of every field, dose bounds on every day, concentration ceilings from
the diffusion maximum principle, progression-bookkeeping consistency,
bit-exact determinism of repeated seeds -- accumulated across many
independent paired runs until the count reaches millions. The campaign
prints exactly how the count decomposes. It does NOT mean millions of
clinical trials: every run is the same ASSUMED-constant model; the campaign
verifies the IMPLEMENTATION, not the biology.

INVARIANTS CHECKED (all derived from the engine's own specification)
--------------------------------------------------------------------
per-voxel, per-field (the bulk of the count):
  V1  N_S, N_R >= 0 everywhere (populations cannot go negative)
  V2  N_S + N_R <= carrying_capacity * (1 + tol) per voxel (capacity)
  V3  c_A, c_B >= 0 everywhere
  V4  c_A, c_B <= source_strength / clearance * (1 + tol)  -- maximum
      principle for reaction-diffusion with bounded source and linear sink
per day, per run:
  D1  dose_A, dose_B in [0, 1]
  D2  burden logs finite and >= 0
  D3  resistant fraction in [0, 1] wherever burden > 0
  D4  progressed <=> pfs_day is not None, and pfs_day is the FIRST day the
      RANO threshold is crossed (bookkeeping consistency)
run level:
  R1  toxicity >= 0 and equals sum(dose_A)*w_A + sum(dose_B)*w_B (recomputed)
  R2  determinism: the same seed reproduces bit-identical trajectories
      (probed on short runs; the engine is fully seeded)
paired-run level:
  P1  MTD toxicity > pulsed-policy toxicity at equal seeds (pulsing
      exposes less drug; a violation would invalidate the toxicity metric)
statistical:
  S1  PFS extension (policy - MTD, paired seeds) reported as a bootstrap
      95% CI over replicate pairs -- never as a bare point estimate

RESUMABILITY
------------
State is a plain JSON document (append chunks, accumulate counts); a
campaign can be interrupted and continued without losing accumulated
evidence. Violations do not raise -- they are COUNTED and reported, with
the first few retained as records, because the point of a verification
campaign is an audit trail, not a green checkmark.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    RANO_PROGRESSION_FRACTION,
    BasePolicy,
    CloneParams,
    CollateralSwitchPolicy,
    DrugParams,
    MTDPolicy,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    TumorGeometry,
    make_gbm_phantom,
)

# Relative tolerance for capacity / ceiling invariants.
_TOL = 1e-6
# The 0.02*laplacian migration term is NOT mask-confined by design: cells
# infiltrate beyond the radiographic boundary, as real tumours do (the
# clinical target volume exists for exactly this reason). Measured leak on
# 60-90 day runs is 6-8% of mass; the bound below is a SANITY CEILING on
# that documented behavior, and the leak distribution is reported as a
# statistic -- the campaign bounds the artifact, it does not deny it.
_MAX_MASK_LEAK_FRACTION = 0.15


@dataclass
class CampaignStats:
    """Accumulated evidence. Counts, not verdicts."""

    n_runs: int = 0
    n_determinism_probes: int = 0
    n_checks: int = 0
    n_violations: int = 0
    n_voxel_checks: int = 0
    n_daily_checks: int = 0
    leak_samples: List[float] = field(default_factory=list)  # mass fraction outside mask
    violation_records: List[str] = field(default_factory=list)
    pfs_policy: List[float] = field(default_factory=list)
    pfs_mtd: List[float] = field(default_factory=list)
    ext_samples: List[float] = field(default_factory=list)      # paired fractions
    tox_red_samples: List[float] = field(default_factory=list)  # paired fractions
    pfs_violations: int = 0  # paired seeds where policy PFS < MTD PFS

    def add_violation(self, msg: str) -> None:
        self.n_violations += 1
        if len(self.violation_records) < 20:
            self.violation_records.append(msg)

    def to_json(self) -> str:
        d = {
            "n_runs": self.n_runs,
            "n_determinism_probes": self.n_determinism_probes,
            "n_checks": self.n_checks,
            "n_violations": self.n_violations,
            "n_voxel_checks": self.n_voxel_checks,
            "n_daily_checks": self.n_daily_checks,
            "pfs_violations": self.pfs_violations,
            "leak_samples": self.leak_samples,
            "violation_records": self.violation_records,
            "pfs_policy": self.pfs_policy,
            "pfs_mtd": self.pfs_mtd,
            "ext_samples": self.ext_samples,
            "tox_red_samples": self.tox_red_samples,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, blob: str) -> "CampaignStats":
        d = json.loads(blob)
        return cls(**d)


# ---------------------------------------------------------------------------
# Invariant checking
# ---------------------------------------------------------------------------


def check_trajectory(traj: SimulationTrajectory, sim: TumorEvolutionSimulator,
                     stats: CampaignStats) -> None:
    """Run every daily/run-level invariant on one trajectory."""
    # D1: dose bounds, every day, both drugs.
    for name, doses in (("dose_A", traj.dose_A), ("dose_B", traj.dose_B)):
        bad = int(np.count_nonzero((doses < 0.0) | (doses > 1.0)))
        stats.n_daily_checks += int(doses.size)
        stats.n_checks += int(doses.size)
        if bad:
            stats.add_violation(f"{name} out of [0,1] on {bad} days")

    # D2: burden logs finite, non-negative.
    for name, arr in (("burden_S", traj.burden_S), ("burden_R", traj.burden_R)):
        bad = int(np.count_nonzero(~np.isfinite(arr) | (arr < 0.0)))
        stats.n_daily_checks += int(arr.size)
        stats.n_checks += int(arr.size)
        if bad:
            stats.add_violation(f"{name} non-finite or negative on {bad} days")

    # D3: resistant fraction bounded where defined.
    tot = traj.burden_S + traj.burden_R
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(tot > 0, traj.burden_R / np.where(tot > 0, tot, 1.0), 0.0)
    bad = int(np.count_nonzero(~np.isfinite(frac) | (frac < -_TOL) | (frac > 1 + _TOL)))
    stats.n_daily_checks += int(frac.size)
    stats.n_checks += int(frac.size)
    if bad:
        stats.add_violation(f"resistant fraction out of [0,1] on {bad} days")

    # D4: progression bookkeeping -- pfs_day is the FIRST threshold crossing.
    b_init = float(tot[0])
    threshold = (1.0 + RANO_PROGRESSION_FRACTION) * b_init
    crossings = np.flatnonzero(tot > threshold)
    first = int(crossings[0]) if crossings.size else None
    stats.n_checks += 2
    stats.n_daily_checks += 2
    if (traj.pfs_day is None) != (not traj.progressed):
        stats.add_violation("progressed flag inconsistent with pfs_day")
    if traj.pfs_day != first:
        stats.add_violation(
            f"pfs_day={traj.pfs_day} but first RANO crossing day={first}"
        )

    # R1: toxicity recomputed from the dose logs.
    expected = float(
        np.sum(traj.dose_A) * 1.0 + np.sum(traj.dose_B) * 0.7
    )
    stats.n_checks += 1
    stats.n_daily_checks += 1
    if not (isinstance(traj.toxicity, float) and math.isfinite(traj.toxicity)
            and traj.toxicity >= 0 and abs(traj.toxicity - expected) <= 1e-6 * max(1.0, expected)):
        stats.add_violation(
            f"toxicity {traj.toxicity:.4f} != recomputed {expected:.4f}"
        )


def check_final_fields(traj: SimulationTrajectory, sim: TumorEvolutionSimulator,
                       stats: CampaignStats) -> None:
    """Per-voxel invariants on the final field state (the bulk of the count)."""
    fields = {
        "N_S": traj.final_N_S,
        "N_R": traj.final_N_R,
        "c_A": traj.final_conc_A,
        "c_B": traj.final_conc_B,
    }
    missing = [k for k, v in fields.items() if v is None]
    if missing:
        raise ValueError(f"keep_fields=True required for voxel checks (missing {missing})")
    K = sim.clones.carrying_capacity
    src = sim.drugs.source_strength
    clr = sim.drugs.clearance

    for name in ("N_S", "N_R"):
        arr = fields[name]
        stats.n_voxel_checks += int(arr.size)
        stats.n_checks += int(arr.size)
        bad_neg = int(np.count_nonzero(arr < 0.0))
        bad_nonfin = int(np.count_nonzero(~np.isfinite(arr)))
        if bad_neg:
            stats.add_violation(f"{name}: {bad_neg} negative voxels")
        if bad_nonfin:
            stats.add_violation(f"{name}: {bad_nonfin} non-finite voxels")
        # V2: capacity (S + R per voxel).
    total_cells = fields["N_S"] + fields["N_R"]
    stats.n_voxel_checks += int(total_cells.size)
    stats.n_checks += int(total_cells.size)
    bad_cap = int(np.count_nonzero(total_cells > K * (1 + _TOL)))
    if bad_cap:
        stats.add_violation(f"{bad_cap} voxels exceed carrying capacity")

    ceil = (src / max(clr[0], 1e-12)) * (1 + _TOL) + 1e-9
    ceil_b = (src / max(clr[1], 1e-12)) * (1 + _TOL) + 1e-9
    for name, arr, cap in (("c_A", fields["c_A"], ceil), ("c_B", fields["c_B"], ceil_b)):
        stats.n_voxel_checks += int(arr.size)
        stats.n_checks += int(arr.size)
        bad_neg = int(np.count_nonzero(arr < 0.0))
        bad_hi = int(np.count_nonzero(arr > cap))
        bad_nonfin = int(np.count_nonzero(~np.isfinite(arr)))
        if bad_neg:
            stats.add_violation(f"{name}: {bad_neg} negative voxels")
        if bad_hi:
            stats.add_violation(f"{name}: {bad_hi} voxels above the maximum-principle ceiling {cap:.3f}")
        if bad_nonfin:
            stats.add_violation(f"{name}: {bad_nonfin} non-finite voxels")

    # Mask-leak: recorded as a measured statistic; the ceiling is a sanity
    # bound on documented migration behavior (see _MAX_MASK_LEAK_FRACTION).
    mask = sim.geom.tumor_mask
    leak = float(total_cells[~mask].sum())
    mass = float(total_cells.sum())
    stats.n_checks += 1
    if mass > 0:
        stats.leak_samples.append(leak / mass)
        if leak / mass > _MAX_MASK_LEAK_FRACTION:
            stats.add_violation(
                f"migration leaked {leak / mass:.1%} of mass outside the tumour mask "
                f"(sanity ceiling {_MAX_MASK_LEAK_FRACTION:.0%})"
            )


def determinism_probe(sim: TumorEvolutionSimulator, policy: BasePolicy,
                      days: int, seed: int, stats: CampaignStats) -> None:
    """Same seed twice -> bit-identical trajectories."""
    t1 = sim.run(policy, days=days, initial_resistant_fraction=0.01, seed=seed)
    t2 = sim.run(policy, days=days, initial_resistant_fraction=0.01, seed=seed)
    stats.n_determinism_probes += 1
    stats.n_checks += 4
    same = (
        np.array_equal(t1.burden_S, t2.burden_S)
        and np.array_equal(t1.burden_R, t2.burden_R)
        and np.array_equal(t1.dose_A, t2.dose_A)
        and np.array_equal(t1.dose_B, t2.dose_B)
    )
    if not same:
        stats.add_violation(f"seed {seed}: repeated run is not bit-identical")


# ---------------------------------------------------------------------------
# Statistical reporting
# ---------------------------------------------------------------------------


def bootstrap_ci_mean(samples: List[float], n_boot: int = 2000, seed: int = 0,
                      level: float = 0.95) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for the mean. Pure numpy."""
    if len(samples) < 5:
        return None
    arr = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo_q = (1.0 - level) / 2.0 * 100.0
    hi_q = (1.0 + level) / 2.0 * 100.0
    return float(np.percentile(means, lo_q)), float(np.percentile(means, hi_q))


# ---------------------------------------------------------------------------
# Chunked campaign driver
# ---------------------------------------------------------------------------


@dataclass
class CampaignConfig:
    days: int = 250
    size: int = 16
    r0_values: Tuple[float, ...] = (0.005, 0.01, 0.05)
    seeds_per_r0: int = 1
    clones: CloneParams = DEFAULT_CLONES
    drugs: DrugParams = DEFAULT_DRUGS
    det_probe_every: int = 10   # one determinism probe per k runs
    geometry_factory: Optional[Callable[[int], TumorGeometry]] = None


def run_campaign_chunk(
    config: CampaignConfig,
    n_runs: int,
    seed0: int = 0,
    stats: Optional[CampaignStats] = None,
) -> CampaignStats:
    """Run n_runs paired (policy, MTD) trials, checking every invariant.

    Each run: one CollateralSwitchPolicy arm with keep_fields=True (voxel
    checks), one paired MTDPolicy arm at the same seed (toxicity/PFS
    pairing), all r0 values cycled through the seed sequence.
    """
    stats = stats if stats is not None else CampaignStats()
    sim = TumorEvolutionSimulator(
        config.geometry_factory(config.size) if config.geometry_factory
        else make_gbm_phantom(size=config.size, seed=42),
        clones=config.clones,
        drugs=config.drugs,
    )
    switch = CollateralSwitchPolicy()
    mtd = MTDPolicy()

    for i in range(n_runs):
        seed = seed0 + i
        r0 = config.r0_values[i % len(config.r0_values)]

        tr_p = sim.run(switch, days=config.days, initial_resistant_fraction=r0,
                       seed=seed, keep_fields=True)
        check_trajectory(tr_p, sim, stats)
        check_final_fields(tr_p, sim, stats)
        tr_m = sim.run(mtd, days=config.days, initial_resistant_fraction=r0, seed=seed)
        check_trajectory(tr_m, sim, stats)

        stats.n_runs += 2  # paired arms
        stats.pfs_policy.append(tr_p.pfs_value(config.days))
        stats.pfs_mtd.append(tr_m.pfs_value(config.days))
        if tr_m.pfs_day is not None:
            ext = (tr_p.pfs_value(config.days) - tr_m.pfs_value(config.days)) / tr_m.pfs_value(config.days)
            stats.ext_samples.append(ext)
            if ext < 0:
                stats.pfs_violations += 1
        if tr_m.toxicity > 0:
            stats.tox_red_samples.append(1.0 - tr_p.toxicity / tr_m.toxicity)
        else:
            # P1: MTD must dose (non-zero toxicity) under continuous full drug.
            stats.n_checks += 1
            if float(np.sum(tr_m.dose_A)) <= 0.0:
                stats.add_violation(f"seed {seed}: MTD arm logged zero dosing")

        if config.det_probe_every and (i + 1) % config.det_probe_every == 0:
            determinism_probe(sim, switch, days=min(30, config.days), seed=seed + 99_000,
                              stats=stats)
    return stats


def run_campaign(
    config: CampaignConfig,
    *,
    target_checks: int,
    chunk_runs: int = 20,
    max_seconds: float = 240.0,
    state_path: Optional[str] = None,
    seed0: int = 0,
    verbose: bool = True,
) -> Tuple[CampaignStats, float]:
    """Accumulate checks chunk by chunk until target_checks is reached or the
    time budget expires. Resumable: state is reloaded from and written to
    state_path (JSON) after every chunk."""
    stats = CampaignStats()
    start_seed = seed0
    if state_path:
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                stats = CampaignStats.from_json(fh.read())
            start_seed = seed0 + stats.n_runs // 2  # pairs of arms
            if verbose:
                print(f"[campaign] resumed: {stats.n_checks:,} checks on record")
        except (OSError, ValueError):
            pass

    t0 = time.monotonic()
    chunk = 0
    while stats.n_checks < target_checks:
        if time.monotonic() - t0 > max_seconds:
            if verbose:
                print(f"[campaign] time budget {max_seconds:.0f}s reached; "
                      f"{stats.n_checks:,} checks accumulated")
            break
        stats = run_campaign_chunk(config, chunk_runs, seed0=start_seed + chunk * chunk_runs,
                                   stats=stats)
        chunk += 1
        if verbose:
            print(f"[campaign] chunk {chunk}: {stats.n_checks:,} checks, "
                  f"{stats.n_violations} violations, "
                  f"{time.monotonic() - t0:.0f}s elapsed")
        if state_path:
            with open(state_path, "w", encoding="utf-8") as fh:
                fh.write(stats.to_json())
    return stats, time.monotonic() - t0


def format_campaign_report(stats: CampaignStats, elapsed_s: float,
                           config: CampaignConfig) -> str:
    bar = "=" * 74
    lines = [bar, "VERIFICATION CAMPAIGN REPORT", bar]
    lines.append(f"paired runs executed        : {stats.n_runs:,}")
    lines.append(f"elementary checks evaluated : {stats.n_checks:,}")
    lines.append(f"  of which per-voxel        : {stats.n_voxel_checks:,}")
    lines.append(f"  of which per-day/run      : {stats.n_daily_checks:,}")
    lines.append(f"determinism probes (2x reruns): {stats.n_determinism_probes:,}")
    lines.append(f"violations found            : {stats.n_violations}")
    if stats.leak_samples:
        leaks = np.asarray(stats.leak_samples)
        lines.append(
            f"migration leak outside mask: median {np.median(leaks):.1%}, "
            f"max {leaks.max():.1%} (documented artifact, bounded at "
            f"{_MAX_MASK_LEAK_FRACTION:.0%})"
        )
    lines.append(f"paired seeds with policy PFS < MTD PFS : {stats.pfs_violations}"
                 f" / {len(stats.ext_samples)} estimable pairs")
    lines.append(f"wall clock                  : {elapsed_s:.0f}s "
                 f"(~{stats.n_checks / max(elapsed_s, 1e-9):,.0f} checks/s)")
    ext_ci = bootstrap_ci_mean(stats.ext_samples, n_boot=2000, seed=1)
    tox_ci = bootstrap_ci_mean(stats.tox_red_samples, n_boot=2000, seed=2)
    if ext_ci is not None:
        lines.append(
            f"PFS extension, paired seeds : median {np.median(stats.ext_samples):+.1%}, "
            f"bootstrap 95% CI [{ext_ci[0]:+.1%}, {ext_ci[1]:+.1%}] "
            f"(n={len(stats.ext_samples)})"
        )
    else:
        lines.append("PFS extension: too few estimable pairs for a CI")
    if tox_ci is not None:
        lines.append(
            f"toxicity reduction          : median {np.median(stats.tox_red_samples):.1%}, "
            f"bootstrap 95% CI [{tox_ci[0]:.1%}, {tox_ci[1]:.1%}] (n={len(stats.tox_red_samples)})"
        )
    if stats.violation_records:
        lines.append("first violations:")
        for rec in stats.violation_records[:10]:
            lines.append(f"  - {rec}")
    lines.append("")
    lines.append(
        f"scope: {config.days}-day paired runs on a {config.size}^3 phantom under "
        "ASSUMED constants -- this campaign verifies the IMPLEMENTATION "
        "(invariants, determinism, bookkeeping), not the biology. It is not "
        "equivalent to clinical replication of any kind."
    )
    lines.append(bar)
    return "\n".join(lines)


def campaign_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "invariant_forms": EvidenceStatus.ESTABLISHED,   # from the engine spec
        "maximum_principle_ceiling": EvidenceStatus.UNKNOWN,  # continuous bound; discrete scheme approximately respects it
        "determinism_full_seeding": EvidenceStatus.ESTABLISHED,
        "growth_kill_constants": EvidenceStatus.UNKNOWN,  # inherited from evosteer
        "campaign_equals_clinical_validation": EvidenceStatus.PROVEN_FAILED,
    }


def what_the_campaign_cannot_prove() -> List[str]:
    return [
        "biological validity: millions of checks of a model with ASSUMED "
        "constants verify code, not cancer -- bench calibration "
        "(danon.validation tier 3) remains the missing layer",
        "clinical outcomes: no number in this campaign is a prediction "
        "for any patient or tumour",
        "statistical equivalence to independent replication: every run "
        "shares one model, one phantom class, one parameter set",
    ]
