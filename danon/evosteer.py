"""
EvoSteer: spatial evolutionary treatment scheduling for resistant sub-clones.

WHAT PROBLEM THIS SOLVES
------------------------
Maximum-tolerated-dose (MTD) protocols apply continuous therapeutic pressure
and select for resistant sub-clones. Evolutionary steering is the alternative:
modulate drug pressure so sensitive cells keep competing against resistant
ones, or switch drugs when a resistant clone rises, exploiting the fitness
cost of resistance (collateral sensitivity). This module simulates that
dynamics in a 3D spatial tumour and compares steering policies against MTD.

TWO DISTINCT STRATEGIES, NOT ONE
--------------------------------
The literature -- and most project abstracts -- conflate two policies:

1. ADAPTIVE DOSE MODULATION (Gatenby): keep using Drug A, but only when tumour
   burden exceeds its baseline; withdraw when it falls back. Sensitive cells
   survive holidays and suppress resistant ones through competition.
   Clinically anchored: in the abiraterone adaptive-therapy phase 2 trial in
   metastatic castration-resistant prostate cancer, median time to progression
   was 33.5 months under adaptive dosing vs 14.3 months under standard of care
   (ratio 2.34; Gallagher et al., JAMA Oncology; Zhang et al., eLife 2022).

2. COLLATERAL SWITCHING: use Drug A until the resistant fraction crosses a
   threshold, then switch to Drug B that kills the resistant clone (which paid
   a fitness cost for A-resistance). The collateral-sensitivity network is
   REAL but CONTINGENT: demonstrated by evolutionary steering in lung cancer
   (Acar et al. 2020), yet some contexts show cross-resistance instead
   (e.g. soft-tissue sarcoma mTKI pairs, Elms et al. 2025). Here the network
   is an explicit parameter set, and the module demonstrates the no-collateral
   scenario where switching loses its edge.

BOTH ARE SIMULATED, ALONGSIDE MTD, IN ONE ENGINE.

WHAT IS ANCHORED, WHAT IS ASSUMED
---------------------------------
LITERATURE anchors used for calibration and endpoints:
  - adaptive/standard TTP ratio 2.34 (mCRPC abiraterone trial)
  - the documented FAILURE regime: adaptive therapy with a very low initial
    resistant fraction can progress EARLIER than standard care
    (Brady-Nicholls et al. 2022, non-spatial ODE game models). A steering
    model that only ever shows steering winning is suspect -- this
    implementation ATTEMPTED the reproduction and could not achieve it (see
    CALIBRATION below); the attempt and its outcome are reported, not hidden.
  - progression threshold: RANO criteria define progression as >=25% increase
    in enhancing lesion size. PFS in this module is measured at that fraction
    of baseline burden.

ASSUMED (bench-measurable, see danon.validation tier 3): every growth rate,
cost of resistance, kill constant and EC50, diffusion and clearance
coefficient, and the GBM parameter set as a whole. The kill constants are
exactly what a CCK-8/IC50 bench campaign measures; the diffusion/clearance
constants are what a spheroid drug-penetration assay measures. Until then the
dynamics CLASS is calibrated (see _calibrate note below), not the numbers.

CALIBRATION (what reproduced, what did not)
-------------------------------------------
The simulator's constants were adjusted against the anchors, with mixed
results -- reported in full, because a rigged model only ever reports wins:
  (a) REPRODUCED: continuous MTD drives resistant-fraction dominance
      mid-trial (dominance around day 20-30 at the default configuration).
  (b) DIRECTION REPRODUCED, MAGNITUDE NOT: adaptive dosing extends median
      PFS over MTD by ~1.3-1.6x in this implementation, below the published
      2.34 anchor. The gap is a model-class property: at partial resistance
      R retains net growth under full Drug-A pressure, so MTD never achieves
      the durable cytoreduction that would make steering look better.
  (c) NOT REPRODUCED: the low-initial-resistant-fraction failure regime.
      In this spatial model MTD cannot durably control the tumour at ANY
      initial fraction -- sanctuary reservoirs guarantee regrowth, and
      raising R's Drug-A kill above R's growth rate collapses BOTH arms to
      long-term control. The intermediate window (MTD controls, adaptive
      does not) was not found in the swept parameter space. The regime was
      established in non-spatial ODE models where continuous therapy can
      cytoreduce the whole compartment; spatial sanctuaries plausibly
      abolish it, but this model neither reproduces nor refutes that.
Constants are marked CALIBRATED below and are a property of THIS
implementation -- re-run the calibration when any assumption changes.

CORRECTION (anchor (c) re-opened: the model was wrong, not just unlucky)
-----------------------------------------------------------------------
Anchor (c) above was investigated by falsification rather than by sweeping:
if resistance is a property of the mechanism, then removing the mechanism
must remove the resistance. It did not. With mutation_rate = 0 AND no seeded
clone, resistance still reached R/(S+R) = 1.000 -- so mutation was doing no
work at all, and mu = 0 and mu = 5e-5 produced identical PFS and dominance
days. Two defects were responsible:
  (i) NO EXTINCTION. Densities are continuous, so a remnant of 10^-9 of a
      voxel (~0.000005 of one cell) grew deterministically back to dominance.
      Every MTD arm failed at every dose because nothing could ever die out.
      Fixed by MIN_VIABLE_DENSITY: a voxel holding under half a cell is
      zeroed at end of day, so eradication is exact.
  (ii) A DEGENERATE RATIO. R/(S+R) reaches 1.0 the moment the sensitive
      population is driven extinct, regardless of whether anything survives;
      dominance was declared at a total burden of 0.016 cells, and the
      collateral-switch policy ACTED on that number. Fixed by
      DETECTION_BURDEN_DENSITY: below ~1e8 cells (~1 cm^3) the fraction is
      reported as 0.0 and no policy may switch on it.
RE-MEASURED after the fix (same geometry, seed 101, 365 days):
  - Pre-existing resistance (initial fraction 0.005, the calibrated default):
    MTD PFS ~165-176 d across seeds 0-5 with dominance ~day 25; the
    collateral switch never reaches progression and ends below the detection
    floor. The steering advantage in (b) survives -- but see below.
  - NO pre-existing resistant clone (fraction 0.0, mu = 5e-5): BOTH arms
    eradicate the tumour (final burden exactly 0). Continuous MTD is
    sufficient and the adaptive policy buys nothing.
  - Resistance from mutation alone needs mu ~ 5e-3 (100x the calibrated
    default) to emerge within 365 days.
The honest conclusion is therefore NOT 'MTD always fails'. It is that the
engine's qualitative verdict is set by an ASSUMED input -- whether a
resistant subclone exists at diagnosis -- which this project does not
measure (the default comes from a calibrated transcriptome mapping, not a
measured clonal fraction). A model whose conclusion flips on an unmeasured
input cannot support a clinical claim, and does not make one.

SCOPE -- deliberately excluded
------------------------------
- No DICOM/MONAI/SegResNet segmentation: the simulator consumes a tumour mask,
  proliferation map and vessel map (see make_gbm_phantom for the contract).
  Segmentation is an upstream preprocessing stage with its own validated
  models; coupling it here would add a heavy dependency without adding a
  testable claim. A synthetic phantom is provided so the pipeline runs
  end-to-end.
- No VR/OpenXR visualisation: the spatial fields are numpy arrays; a 3D drug
  gradient renders identically in any viewer and real-time frame rate is a
  presentation property, not a scientific one.
- No immune compartment, no angiogenesis dynamics, no spatial mutation of a
  third double-resistant clone: each is a real omission, listed in
  what_this_model_cannot_know().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------
# Literature anchors
# ---------------------------------------------------------------------------

#: Median time to progression (months): adaptive vs standard of care,
#: abiraterone adaptive-therapy phase 2, mCRPC (Gallagher et al., JAMA Oncol;
#: Zhang et al., eLife 2022 report the same trial's modelling).
ADAPTIVE_TTP_MONTHS_MCRPC: Tuple[float, float] = (33.5, 14.3)
#: 33.5 / 14.3 = 2.34. The dynamics-class calibration target for the
#: adaptive arm's advantage in the DEFAULT regime.
ADAPTIVE_TTP_RATIO_ANCHOR: float = ADAPTIVE_TTP_MONTHS_MCRPC[0] / ADAPTIVE_TTP_MONTHS_MCRPC[1]

#: RANO response criteria: progression = >=25% increase in enhancing lesion.
RANO_PROGRESSION_FRACTION = 0.25

#: Cells per voxel at unit density (N = 1) on the engine's real geometry. A
#: 3.667 mm engine voxel is 49.3 mm^3 = 0.0493 cm^3, and solid tumour runs at
#: roughly 1e8 cells/cm^3, so a saturated voxel holds ~4.9e6 cells. This
#: constant exists to convert the continuum density into a CELL COUNT for two
#: purposes only -- the extinction floor and the detection floor below. It
#: never rescales the dynamics, which stay in normalised density units.
CELLS_PER_VOXEL_AT_UNIT_DENSITY = 4.93e6

#: A population smaller than one cell is not a population. Any voxel holding
#: less than half a cell is zeroed at end of day. Without this floor a
#: sub-single-cell remnant (10^-9 of a voxel) grew deterministically back to
#: dominance, so essentially every MTD arm failed at every dose and at EVERY
#: mutation rate -- including mutation_rate = 0, which is what exposed the
#: defect: resistance was pre-seeded and mutation was doing no work at all.
MIN_VIABLE_CELLS_PER_VOXEL = 0.5
MIN_VIABLE_DENSITY = MIN_VIABLE_CELLS_PER_VOXEL / CELLS_PER_VOXEL_AT_UNIT_DENSITY

#: Disease below about this burden is not a detectable lesion, so R/(S+R) is
#: not a clinical quantity. 1e8 cells is ~1 cm^3 of tumour at the density
#: above. Below the floor the resistant fraction is reported as 0.0 and no
#: policy may act on it: otherwise a tumour that has been driven to extinction
#: except for a few resistant cells reports R-fraction 1.00, which reads as
#: 'resistance took over' when the honest statement is 'the sensitive
#: population is gone and there is almost nothing left of the tumour'.
DETECTION_BURDEN_CELLS = 1.0e8
DETECTION_BURDEN_DENSITY = DETECTION_BURDEN_CELLS / CELLS_PER_VOXEL_AT_UNIT_DENSITY

#: Observed adaptive/MTD PFS-ratio range for THIS implementation across the
#: calibrated configuration and seed variation (DERIVED from the calibration
#: runs). The published 2.34 anchor sits OUTSIDE this band: this model class
#: understates the adaptive advantage -- see CALIBRATION in the docstring.
MEASURED_ADAPTIVE_MTD_RATIO_BAND: Tuple[float, float] = (1.2, 1.8)


# ---------------------------------------------------------------------------
# Geometry: tumour mask, proliferation map, vessel map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TumorGeometry:
    """Spatial substrate the simulator consumes.

    tumor_mask   : bool array, True inside tumour.
    proliferative: float array in [0, 1], relative growth rate per voxel
                   (necrotic/hypoxic core -> low, rim -> high).
    vessels      : float array in [0, 1], drug source strength per voxel.
    voxel_mm     : edge length of one voxel in mm (bookkeeping only).
    """

    tumor_mask: np.ndarray
    proliferative: np.ndarray
    vessels: np.ndarray
    voxel_mm: float = 1.0

    def __post_init__(self) -> None:
        for name in ("tumor_mask", "proliferative", "vessels"):
            arr = getattr(self, name)
            if arr.shape != self.tumor_mask.shape or arr.ndim != 3:
                raise ValueError(f"{name} must be a 3D array matching tumor_mask")
        if self.tumor_mask.sum() == 0:
            raise ValueError("empty tumour mask")
        if self.vessels.max() == 0:
            raise ValueError("no vasculature: no drug can reach the tumour")


def make_gbm_phantom(
    size: int = 24,
    seed: int = 0,
    radius_frac: float = 0.38,
    core_frac: float = 0.55,
    n_vessels: int = 5,
    rim_vessel: bool = True,
) -> TumorGeometry:
    """Synthetic GBM-like substrate with diffusion sanctuaries.

    Sphere tumour with a ruffled boundary; a hypoxic/necrotic core with
    suppressed proliferation; penetrating vessels (random chords) plus an
    optional peripheral shell. Regions of the core far from every vessel are
    the sanctuary pockets: drug arrives by slow diffusion only, which is the
    spatial phenomenon that lets resistant clones persist under MTD.

    A real case replaces this with the outputs of a segmentation model run on
    paired MRI: tumour/edema mask, enhancing-rim map, and perfusion-derived
    vessel map. The contract is TumorGeometry either way.
    """
    if size < 8:
        raise ValueError("phantom too small; use size >= 8")
    rng = np.random.default_rng(seed)
    idx = np.arange(size)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    c = (size - 1) / 2.0
    r = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2)
    radius = radius_frac * size

    # Ruffled boundary: radial noise perturbation.
    wobble = 0.08 * radius * np.sin(0.9 * X + 1.7 * Y) * np.cos(0.8 * Z + 0.6 * Y)
    tumor_mask = r + wobble < radius

    proliferative = np.where(
        tumor_mask,
        np.where(r < core_frac * radius, 0.20, 1.0),
        0.0,
    )

    vessels = np.zeros((size, size, size), dtype=float)
    # Penetrating chords: random lines through the centre region.
    for _ in range(n_vessels):
        p0 = c + rng.normal(0, radius * 0.35, 3)
        direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        t_line = np.linspace(-radius * 1.6, radius * 1.6, 240)
        pts = p0[None, :] + t_line[:, None] * direction[None, :]
        for pt in pts:
            xi, yi, zi = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
            if 0 <= xi < size and 0 <= yi < size and 0 <= zi < size:
                vessels[xi, yi, zi] = 1.0
    if rim_vessel:
        rim = tumor_mask & (r + wobble > 0.85 * radius)
        vessels[rim] = np.maximum(vessels[rim], 0.7)

    return TumorGeometry(
        tumor_mask=tumor_mask,
        proliferative=proliferative,
        vessels=vessels,
        voxel_mm=1.0,
    )


# ---------------------------------------------------------------------------
# Biology and pharmacology parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloneParams:
    """Per-clone growth parameters (Lotka-Volterra competition).

    growth_S and growth_R are per-day logistic rates; cost_of_resistance is
    the fitness price of A-resistance expressed in the R growth rate (this is
    the trade-off collateral steering exploits).

    competition is the asymmetric inter-clone competition coefficient: R
    feels the FULL density of S (sensitive cells crowd it out during drug
    holidays -- Gatenby's mechanism), while S feels R only at `competition`
    weight (an R cell that paid a growth cost competes weakly). competition
    < 1 is what makes holiday-phase suppression possible; with competition
    = 1 the model degenerates to shared-capacity competition.

    drug_sensitivity_scale_R multiplies R's Drug-A kill rate relative to the
    resistance profile (values far below 1 = strong resistance; moderate
    values = partial resistance, so MTD still prunes R in well-perfused
    regions and vasculature geometry, not the clone label alone, decides
    where R survives).
    """

    growth_S: float          # 1/day, CALIBRATED
    growth_R: float          # 1/day, CALIBRATED (= growth_S * (1 - cost))
    carrying_capacity: float = 1.0   # per-voxel density ceiling
    mutation_rate: float = 0.0       # S -> R per S division
    competition: float = 0.6         # S-on-R crowding weight, CALIBRATED
    drug_sensitivity_scale_R: float = 0.25  # R's A-kill scale, CALIBRATED


@dataclass(frozen=True)
class DrugParams:
    """Saturating kill kinetics and transport for two drugs.

    kill_max[clone] is the max per-day kill rate at saturating concentration;
    EC50 the concentration at half effect; diffusion D (voxel^2/day) and
    clearance (1/day) govern the spatial gradient; toxicity_weight converts
    cumulative dose into a toxicity proxy.
    """

    kill_max_S: Tuple[float, float]   # per drug (A, B), 1/day
    kill_max_R: Tuple[float, float]
    ec50: Tuple[float, float]
    diffusion: Tuple[float, float]
    clearance: Tuple[float, float]
    toxicity_weight: Tuple[float, float] = (1.0, 0.7)
    source_strength: float = 1.0      # conc/day at full-dose vessels


# CALIBRATED parameter set (see module docstring "CALIBRATION").
# Every rate below is ASSUMED biology adjusted until the dynamics class
# reproduced the anchors; re-calibrate on bench measurements.
DEFAULT_CLONES = CloneParams(
    growth_S=0.16,
    growth_R=0.16 * (1.0 - 0.25),
    mutation_rate=5e-5,
    competition=0.6,
    drug_sensitivity_scale_R=0.25,
)
DEFAULT_DRUGS = DrugParams(
    kill_max_S=(0.34, 0.10),
    kill_max_R=(0.02, 0.34),
    ec50=(0.30, 0.30),
    diffusion=(0.35, 0.35),
    clearance=(0.10, 0.10),
)


@dataclass(frozen=True)
class ImmuneParams:
    """Optional T-cell clearance layer (see danon.evotwin for the cohort use).

    kill_max          : max per-day immune kill rate, reached at zero local
                        density (1/day).
    neoantigen_factor : [0, 1] antigenicity scale. GBM is immune-cold (low
                        tumour mutational burden), and checkpoint MONOtherapy
                        failed in recurrent GBM (CheckMate 143: nivolumab
                        9.8 vs bevacizumab 10.0 months, HR 1.04) -- so the
                        default is deliberately weak. This layer models a
                        HYPOTHESISED low-density clearance mechanism, not an
                        evidence-backed GBM therapy.

    Both constants are ASSUMED and bench-measurable (co-culture kill-rate
    assays); the density gate shape is a modelling choice, listed in
    what_this_model_cannot_know().
    """

    kill_max: float = 0.45
    neoantigen_factor: float = 0.30


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass
class PolicyState:
    day: int
    burden: float              # total tumour cell density (summed)
    burden_initial: float
    resistant_fraction: float  # R / (S + R), burden-weighted


class PolicyKind(str, Enum):
    MTD = "mtd"
    ADAPTIVE = "adaptive_dose"
    COLLATERAL_SWITCH = "collateral_switch"


class BasePolicy:
    kind: PolicyKind = PolicyKind.MTD

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        """Return (dose_A, dose_B), each in [0, 1]."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-trial state (called at the start of each run).

        Stateful policies carried across replicates would leak the previous
        replicate's regime into the next one; the runner calls this before
        every trial."""

    def describe(self) -> str:
        return self.kind.value


class MTDPolicy(BasePolicy):
    """Continuous maximum tolerated dose of Drug A."""

    kind = PolicyKind.MTD

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        return 1.0, 0.0


class AdaptiveDosePolicy(BasePolicy):
    """Gatenby-style dose modulation of Drug A.

    Dose at maximum while burden exceeds its initial baseline; withdraw
    entirely when the tumour is pushed back to baseline. Sensitive cells
    regrow during holidays and compete against resistant ones.
    """

    kind = PolicyKind.ADAPTIVE

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        return (1.0, 0.0) if state.burden > state.burden_initial else (0.0, 0.0)


class CollateralSwitchPolicy(BasePolicy):
    """Switch Drug A -> Drug B on resistant-fraction thresholds.

    Hysteresis is required: theta_off must be strictly below theta_on, or the
    policy chatters between drugs day to day.
    """

    kind = PolicyKind.COLLATERAL_SWITCH

    def __init__(self, theta_on: float = 0.15, theta_off: float = 0.05) -> None:
        if not 0.0 < theta_off < theta_on < 1.0:
            raise ValueError(
                f"need 0 < theta_off < theta_on < 1, got on={theta_on}, off={theta_off}"
            )
        self.theta_on = theta_on
        self.theta_off = theta_off
        self._on_b = False

    def reset(self) -> None:
        self._on_b = False

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        if state.resistant_fraction >= self.theta_on:
            self._on_b = True
        elif state.resistant_fraction <= self.theta_off:
            self._on_b = False
        return (0.0, 1.0) if self._on_b else (1.0, 0.0)

    def describe(self) -> str:
        return f"switch(A->B @ R>={self.theta_on:.0%}, back @ R<={self.theta_off:.0%})"


def make_policy(kind: PolicyKind, **kwargs) -> BasePolicy:
    if kind is PolicyKind.MTD:
        return MTDPolicy()
    if kind is PolicyKind.ADAPTIVE:
        return AdaptiveDosePolicy()
    if kind is PolicyKind.COLLATERAL_SWITCH:
        return CollateralSwitchPolicy(**kwargs)
    raise ValueError(f"unknown policy kind: {kind}")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


from danon.anisotropy import LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, divergence_d_grad


def _laplacian(f: np.ndarray) -> np.ndarray:
    """Zero-flux (edge-padded) 3D Laplacian."""
    p = np.pad(f, 1, mode="edge")
    return (
        p[:-2, 1:-1, 1:-1]
        + p[2:, 1:-1, 1:-1]
        + p[1:-1, :-2, 1:-1]
        + p[1:-1, 2:, 1:-1]
        + p[1:-1, 1:-1, :-2]
        + p[1:-1, 1:-1, 2:]
        - 6.0 * f
    )


@dataclass
class SimulationTrajectory:
    policy: str
    days: int
    burden_S: np.ndarray            # per-day total sensitive density
    burden_R: np.ndarray
    dose_A: np.ndarray
    dose_B: np.ndarray
    pfs_day: Optional[int]          # first day burden > (1+RANO)*initial
    progressed: bool
    initial_resistant_fraction: float
    toxicity: float                 # cumulative weighted dose proxy
    final_resistant_fraction: float
    resistant_dominance_day: Optional[int]   # first day R fraction > 0.5
    # Optional final fields for spatial analytics (sanctuary inspection).
    final_conc_A: Optional[np.ndarray] = None
    final_conc_B: Optional[np.ndarray] = None
    final_N_S: Optional[np.ndarray] = None
    final_N_R: Optional[np.ndarray] = None

    def pfs_value(self, trial_days: int) -> float:
        """Censored PFS in days: progression day, else the trial length."""
        return float(self.pfs_day) if self.pfs_day is not None else float(trial_days)


class TumorEvolutionSimulator:
    """Daily-step 3D tumour ABM with diffusing drugs and clonal competition.

    Per day: policy doses, drug fields diffuse/clear (sub-stepped for
    stability), clones grow logistically with local carrying capacity,
    mutate S->R stochastically, are killed by saturating drug kinetics, and
    migrate slowly. Density state is continuous; stochasticity enters through
    the mutation draw and a small multiplicative growth noise, which is the
    dominant variance source (initial clone placement and mutation timing).
    """

    def __init__(
        self,
        geometry: TumorGeometry,
        clones: CloneParams = DEFAULT_CLONES,
        drugs: DrugParams = DEFAULT_DRUGS,
        diffusion_field: Optional[object] = None,
        cell_motility: Optional[float] = None,
    ) -> None:
        self.geom = geometry
        self.clones = clones
        self.drugs = drugs
        self._shape = geometry.tumor_mask.shape
        self._tumor_idx = geometry.tumor_mask
        # Optional spatially heterogeneous transport. The field is reduced to a
        # mean-one relative pattern, so the drug's calibrated mean diffusivity
        # is unchanged and ONLY the spatial variation is added. That keeps the
        # effect of heterogeneity separable from the effect of changing the
        # absolute scale -- conflating the two would make any measured
        # difference unattributable.
        self._diff_pattern: Optional[np.ndarray] = None
        if diffusion_field is not None:
            field = np.asarray(getattr(diffusion_field, "d_engine", diffusion_field), dtype=float)
            if field.shape != self._shape:
                raise ValueError(
                    f"diffusion field shape {field.shape} does not match geometry "
                    f"{self._shape}"
                )
            if np.any(field < 0.0):
                raise ValueError("diffusion field must be non-negative")
            mean = float(field.mean())
            if mean <= 0.0:
                raise ValueError("diffusion field has zero mean: no transport at all")
            self._diff_pattern = field / mean
        # Cell motility in voxel^2/day. None keeps the historical hardcoded
        # value so existing calibration anchors are untouched.
        if cell_motility is not None and cell_motility < 0.0:
            raise ValueError("cell_motility must be non-negative")
        self._cell_motility = cell_motility

    # -- initial conditions -------------------------------------------------

    def seed(
        self,
        initial_resistant_fraction: float,
        seed: int = 0,
        cluster_sigma: float = 2.0,
        fill_fraction: float = 0.65,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Place S everywhere in the tumour and R in one clustered subclone.

        A single spatially clustered resistant seed (not uniform sprinkling)
        is the conservative, clinically-motivated layout: resistance emerges
        in local patches, then competes spatially. Returns (N_S, N_R).

        fill_fraction < 1 seeds the tumour BELOW its spatial carrying
        capacity: a treated tumour at diagnosis is still expanding, and a
        fully saturated initial condition would freeze every clonal dynamic
        (no growth, no competitive release where drug kills S). fill is
        CALIBRATED with the rest of the parameter set.
        """
        if not 0.0 <= initial_resistant_fraction < 1.0:
            raise ValueError("initial_resistant_fraction must be in [0, 1)")
        rng = np.random.default_rng(seed)
        mask = self.geom.tumor_mask
        N_S = mask.astype(float) * self.clones.carrying_capacity * fill_fraction

        # A fraction of exactly 0 means NO resistant clone at diagnosis: any
        # resistance in the trial must then be created by mutation. This is the
        # only configuration in which 'resistance evolves' is a testable
        # statement rather than an initial condition -- the clustered-seed
        # path always places R in at least 9 voxels, which at any nonzero
        # fraction is already a viable population.
        if initial_resistant_fraction == 0.0:
            return N_S, np.zeros_like(N_S)

        voxels = np.argwhere(mask)
        centre = voxels[rng.integers(len(voxels))].astype(float)
        dist2 = np.sum(
            (np.argwhere(mask).astype(float) - centre[None, :]) ** 2, axis=1
        )
        weights = np.exp(-dist2 / (2.0 * cluster_sigma**2))
        # Compact seed: the ~2% highest-weight tumour voxels (min 9), NOT a
        # wide Gaussian skirt. A resistant subclone at diagnosis is a local
        # patch; a wide kernel sprinkles R across half the tumour and dilutes
        # the spatial competition this model exists to study.
        n_cluster = max(9, int(round(0.02 * len(voxels))))
        take = np.argsort(-weights)[:n_cluster]
        cluster_voxels = voxels[take]

        N_R = np.zeros_like(N_S)
        # Scale cluster density so total R equals fraction of total burden.
        total = N_S.sum()
        target_R = initial_resistant_fraction * total
        per_voxel = min(self.clones.carrying_capacity, target_R / len(cluster_voxels))
        for vx, vy, vz in cluster_voxels:
            N_R[vx, vy, vz] = per_voxel
            N_S[vx, vy, vz] = max(0.0, N_S[vx, vy, vz] - per_voxel)
        return N_S, N_R

    # -- one trial -----------------------------------------------------------

    def run(
        self,
        policy: BasePolicy,
        days: int = 365,
        initial_resistant_fraction: float = 0.01,
        seed: int = 0,
        keep_fields: bool = False,
        growth_noise: float = 0.05,
        fill_fraction: float = 0.65,
        immune: Optional[ImmuneParams] = None,
        observation_hook: Optional[Callable[[int, float, float], None]] = None,
        initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        clearance_pattern: Optional[np.ndarray] = None,
        field_hook: Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]] = None,
    ) -> SimulationTrajectory:
        """Run one trial.

        observation_hook, when given, is called at the END of each day with
        (day, burden_S, burden_R) AFTER that day's state update and BEFORE
        the next day's dose decision -- the causal order a clinical
        monitoring loop needs (an assay drawn on day d informs day d+1's
        dosing). It receives aggregate burdens only and must not mutate
        simulator state; policies never see it, so no hidden-state leakage
        is possible through this channel.

        field_hook, when given, is called IMMEDIATELY AFTER observation_hook
        on the same day with (day, N_S, N_R, c_A, c_B) -- the full spatial
        state the aggregates in observation_hook describe. The arrays are
        internal references: a hook that retains them must copy. Mutating
        them is forbidden (it would perturb the run the policies act on);
        a recorder that only reads and copies cannot affect the trajectory.
        IDENTITY: field_hook=None is a no-op -- the trajectory is bit-identical
        to the historical path. A recording hook that only copies also
        leaves the trajectory bit-identical, which the test suite pins by
        comparing burden arrays with and without a recording hook attached.

        initial_state, when given, is an (N_S, N_R) pair that REPLACES the
        clustered seed -- the clinical-twin path where a measured t=0 map
        (e.g. 5-ALA fluorescence) sets the initial tumour density. It must
        match the geometry shape and be non-negative somewhere. When given,
        initial_resistant_fraction is ignored (the fraction is whatever the
        supplied state carries). IDENTITY: initial_state=sim.seed(r, seed=s,
        fill_fraction=f) reproduces the default path exactly (same rng
        consumption: seed() draws from its own generator).

        clearance_pattern, when given, is a per-voxel RELATIVE multiplier on
        both drugs' clearance (mean-normalised to 1.0, like the diffusion
        pattern, so the calibrated mean clearance is unchanged and only the
        spatial variation is added): lambda(x) = lambda_0 * pattern(x). The
        clinical-twin module maps a K-trans map here. IDENTITY: None or an
        all-ones pattern reduces to the historical uniform clearance.
        """
        if days < 2:
            raise ValueError("need at least 2 days")
        if initial_state is not None:
            N_S0, N_R0 = initial_state
            N_S0 = np.asarray(N_S0, dtype=float)
            N_R0 = np.asarray(N_R0, dtype=float)
            if N_S0.shape != self._shape or N_R0.shape != self._shape:
                raise ValueError(
                    f"initial_state shapes {(N_S0.shape, N_R0.shape)} do not "
                    f"match geometry {self._shape}"
                )
            if np.any(N_S0 < 0.0) or np.any(N_R0 < 0.0):
                raise ValueError("initial_state densities must be non-negative")
            if float(N_S0.sum() + N_R0.sum()) <= 0.0:
                raise ValueError("initial_state is empty: no tumour cells at t=0")
        clr_pat_norm: Optional[np.ndarray] = None
        if clearance_pattern is not None:
            clr_pat = np.asarray(clearance_pattern, dtype=float)
            if clr_pat.shape != self._shape:
                raise ValueError(
                    f"clearance pattern shape {clr_pat.shape} does not match "
                    f"geometry {self._shape}"
                )
            if np.any(clr_pat < 0.0):
                raise ValueError("clearance pattern must be non-negative")
            mean = float(clr_pat.mean())
            if mean <= 0.0:
                raise ValueError("clearance pattern has zero mean: no clearance at all")
            clr_pat_norm = clr_pat / mean
        self._run_clr_pattern = clr_pat_norm
        rng = np.random.default_rng(seed)
        policy.reset()  # stateful policies must not leak across replicates
        if initial_state is not None:
            N_S, N_R = N_S0.copy(), N_R0.copy()
        else:
            N_S, N_R = self.seed(
                initial_resistant_fraction, seed=seed, fill_fraction=fill_fraction
            )

        c_A = np.zeros(self._shape)
        c_B = np.zeros(self._shape)
        K = self.clones.carrying_capacity
        rS, rR, mu = (
            self.clones.growth_S,
            self.clones.growth_R,
            self.clones.mutation_rate,
        )
        prolif = self.geom.proliferative
        vessels = self.geom.vessels

        # Diffusion sub-steps for stability (3D explicit: 6*D*dt/dx^2 <= 1).
        # With a heterogeneous field the binding constraint is the LARGEST local
        # D, not the mean, so the pattern's maximum is what sets the step.
        dt = 1.0
        dA, dB = self.drugs.diffusion
        pattern_peak = float(self._diff_pattern.max()) if self._diff_pattern is not None else 1.0
        sub_dt = min(
            0.4 / (6.0 * max(dA * pattern_peak, 1e-9)),
            0.4 / (6.0 * max(dB * pattern_peak, 1e-9)),
            1.0,
        )
        n_sub = max(1, int(math.ceil(dt / sub_dt)))
        dts = dt / n_sub

        b_init = float(N_S.sum() + N_R.sum())
        pfs_threshold = (1.0 + RANO_PROGRESSION_FRACTION) * b_init

        bS = np.empty(days)
        bR = np.empty(days)
        dA_log = np.empty(days)
        dB_log = np.empty(days)
        pfs_day: Optional[int] = None
        dominance_day: Optional[int] = None
        toxicity = 0.0

        kmax_S = self.drugs.kill_max_S
        kmax_R = self.drugs.kill_max_R
        ec50 = self.drugs.ec50
        clr = self.drugs.clearance
        clr_pat = getattr(self, "_run_clr_pattern", None)
        src = self.drugs.source_strength

        for day in range(days):
            burden = float(N_S.sum() + N_R.sum())
            # R/(S+R) is a clinical quantity only above the detection floor.
            # Below it the ratio is degenerate: kill S to extinction and the
            # ratio reads 1.00 however few cells remain, so a policy acting on
            # it would switch drugs because a tumour had been ERADICATED.
            detectable = burden >= DETECTION_BURDEN_DENSITY
            r_frac = (
                float(N_R.sum() / burden) if (burden > 0.0 and detectable) else 0.0
            )
            st = PolicyState(
                day=day, burden=burden, burden_initial=b_init, resistant_fraction=r_frac
            )
            doseA, doseB = policy.dose(st)
            bS[day], bR[day] = float(N_S.sum()), float(N_R.sum())
            dA_log[day], dB_log[day] = doseA, doseB
            toxicity += doseA * self.drugs.toxicity_weight[0] + doseB * self.drugs.toxicity_weight[1]

            if pfs_day is None and burden > pfs_threshold:
                pfs_day = day
            if dominance_day is None and detectable and r_frac > 0.5:
                dominance_day = day

            # Drug transport. With a heterogeneous field the divergence form
            # div(D(x) grad C) is used; the scalar path is D * laplacian(C).
            # These coincide exactly when D is constant, so the two branches are
            # the same model in the homogeneous limit.
            if self._diff_pattern is None:
                for _ in range(n_sub):
                    c_A += dts * (dA * _laplacian(c_A) + doseA * src * vessels
                                  - (clr_pat * clr[0] if clr_pat is not None else clr[0]) * c_A)
                    c_B += dts * (dB * _laplacian(c_B) + doseB * src * vessels
                                  - (clr_pat * clr[1] if clr_pat is not None else clr[1]) * c_B)
                    np.maximum(c_A, 0.0, out=c_A)
                    np.maximum(c_B, 0.0, out=c_B)
            else:
                clrA = clr[0] if clr_pat is None else clr[0] * clr_pat
                clrB = clr[1] if clr_pat is None else clr[1] * clr_pat
                for _ in range(n_sub):
                    c_A += dts * (
                        dA * divergence_d_grad(c_A, self._diff_pattern)
                        + doseA * src * vessels - clrA * c_A
                    )
                    c_B += dts * (
                        dB * divergence_d_grad(c_B, self._diff_pattern)
                        + doseB * src * vessels - clrB * c_B
                    )
                    np.maximum(c_A, 0.0, out=c_A)
                    np.maximum(c_B, 0.0, out=c_B)

            # Saturating per-clone kill rates are computed below with growth
            # (clone-structured Drug-A sensitivity for R).

            # Growth with Lotka-Volterra competition, hypoxic suppression,
            # noise. Kill applies to the STANDING population (pharmacodynamic
            # death), not merely to new growth -- otherwise a tumour at
            # carrying capacity is invisible to the drug and no clonal
            # dynamics occur. Competition is ASYMMETRIC: R feels S at full
            # weight (holiday-phase suppression, Gatenby's mechanism); S
            # feels R at `competition` weight (an R cell that paid a growth
            # cost competes weakly).
            noise = np.clip(
                1.0 + rng.normal(0.0, growth_noise, self._shape), 0.0, None
            )
            room_S = np.clip(
                1.0 - (N_S + self.clones.competition * N_R) / K, 0.0, 1.0
            )
            room_R = np.clip(1.0 - (N_S + N_R) / K, 0.0, 1.0)
            grow_S = rS * N_S * room_S * prolif * noise
            grow_R = rR * N_R * room_R * prolif * noise

            # Stochastic S->R mutation from sensitive divisions.
            if mu > 0:
                new_R = rng.poisson(np.clip(grow_S, 0.0, None) * mu)
            else:
                new_R = 0.0

            # Clone-structured kill: R's Drug-A kill is scaled by its
            # (partial) resistance, so Drug A still prunes R in
            # well-perfused regions -- vasculature geometry, not the clone
            # label alone, decides where R survives.
            kill_R_A = kmax_R[0] * self.clones.drug_sensitivity_scale_R * c_A / (
                c_A + ec50[0]
            )
            kill_S = (
                kmax_S[0] * c_A / (c_A + ec50[0]) + kmax_S[1] * c_B / (c_B + ec50[1])
            )
            kill_R = (
                kill_R_A + kmax_R[1] * c_B / (c_B + ec50[1])
            )

            # Optional T-cell clearance: neoantigen-dependent and gated by
            # LOCAL density -- isolated (low-density) tumour cells are the
            # least physically protected, kill is zero at carrying capacity.
            # The kill is RESISTANCE-BLIND (both clones equally), so unlike
            # Drug A it cannot select for drug resistance.
            if immune is not None:
                density = np.clip((N_S + N_R) / K, 0.0, 1.0)
                imm_kill = immune.kill_max * immune.neoantigen_factor * (1.0 - density)
            else:
                imm_kill = 0.0

            N_S = np.clip(
                N_S + grow_S - N_S * (kill_S + imm_kill) - new_R, 0.0, None
            )
            N_R = np.clip(
                N_R + grow_R - N_R * (kill_R + imm_kill) + new_R, 0.0, None
            )

            # Slow cell migration. The coefficient is named rather than inlined
            # so it can be audited in physical units: the historical literal is
            # `danon.anisotropy.LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY`, which is
            # 0.02 mm^2/day on the phantom but 0.27 mm^2/day on real 3.67 mm
            # geometry, against a literature motility range of 0.001-0.005.
            motility = (
                self._cell_motility
                if self._cell_motility is not None
                else LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY
            )
            if motility:
                N_S = np.clip(N_S + motility * _laplacian(N_S), 0.0, None)
                N_R = np.clip(N_R + motility * _laplacian(N_R), 0.0, None)

            # Extinction floor, applied to the end-of-day state: a voxel
            # holding less than half a cell holds no cells. This is what stops
            # a sub-single-cell remnant from regrowing deterministically --
            # continuous densities have no extinction, and every nonzero
            # population grows.
            np.copyto(N_S, 0.0, where=N_S < MIN_VIABLE_DENSITY)
            np.copyto(N_R, 0.0, where=N_R < MIN_VIABLE_DENSITY)

            # Clinical monitoring hook: end-of-day aggregate burdens.
            if observation_hook is not None:
                observation_hook(day, float(N_S.sum()), float(N_R.sum()))
            # Spatial recording hook: the same end-of-day state, full fields.
            if field_hook is not None:
                field_hook(day, N_S, N_R, c_A, c_B)

        final_burden = float(N_S.sum() + N_R.sum())
        traj = SimulationTrajectory(
            policy=policy.describe(),
            days=days,
            burden_S=bS,
            burden_R=bR,
            dose_A=dA_log,
            dose_B=dB_log,
            pfs_day=pfs_day,
            progressed=pfs_day is not None,
            initial_resistant_fraction=initial_resistant_fraction,
            toxicity=toxicity,
            # Gated on the same detection floor as the daily readout: at the
            # horizon the honest statement about a sub-detectable tumour is
            # carried by the absolute burden arrays, not by this ratio.
            final_resistant_fraction=(
                float(N_R.sum() / final_burden)
                if final_burden >= DETECTION_BURDEN_DENSITY
                else 0.0
            ),
            resistant_dominance_day=dominance_day,
        )
        if keep_fields:
            traj.final_conc_A = c_A
            traj.final_conc_B = c_B
            traj.final_N_S = N_S
            traj.final_N_R = N_R
        return traj


# ---------------------------------------------------------------------------
# Trial harness: policy comparison with Monte Carlo replicates
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    policy: str
    n_replicates: int
    median_pfs_days: float
    pfs_iqr: Tuple[float, float]
    frac_progressed: float
    median_toxicity: float
    toxicity_reduction_vs_mtd: Optional[float]
    median_resistance_dominance_day: Optional[float]
    pfs_values: List[float] = field(default_factory=list)


@dataclass
class TrialReport:
    arms: Dict[str, ArmResult]
    days: int
    n_replicates: int
    initial_resistant_fraction: float
    adaptive_vs_mtd_pfs_ratio: float
    verdict: str
    notes: List[str] = field(default_factory=list)


def run_trial(
    geometry: TumorGeometry,
    arms: Optional[Sequence[BasePolicy]] = None,
    *,
    days: int = 365,
    n_replicates: int = 10,
    initial_resistant_fraction: float = 0.01,
    seed0: int = 0,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
) -> TrialReport:
    """Compare policies over Monte Carlo replicates (one seed per replicate).

    Reports the distribution of censored PFS and the toxicity proxy per arm.
    A single-run PFS gain is never reported: the claim is the median and its
    spread, and the fraction of replicates that progressed at all.
    """
    sim = TumorEvolutionSimulator(geometry, clones=clones, drugs=drugs)
    if arms is None:
        arms = [MTDPolicy(), AdaptiveDosePolicy(), CollateralSwitchPolicy()]

    results: Dict[str, ArmResult] = {}
    pfs_by_arm: Dict[str, float] = {}
    tox_by_arm: Dict[str, float] = {}
    for pol in arms:
        pfs_vals: List[float] = []
        tox_vals: List[float] = []
        dom_days: List[float] = []
        for rep in range(n_replicates):
            traj = sim.run(
                pol,
                days=days,
                initial_resistant_fraction=initial_resistant_fraction,
                seed=seed0 + rep,
            )
            pfs_vals.append(traj.pfs_value(days))
            tox_vals.append(traj.toxicity)
            if traj.resistant_dominance_day is not None:
                dom_days.append(float(traj.resistant_dominance_day))
        name = pol.describe()
        arr = np.asarray(pfs_vals)
        med = float(np.median(arr))
        q1, q3 = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
        results[name] = ArmResult(
            policy=name,
            n_replicates=n_replicates,
            median_pfs_days=med,
            pfs_iqr=(q1, q3),
            frac_progressed=float(np.mean([v < days for v in pfs_vals])),
            median_toxicity=float(np.median(tox_vals)),
            toxicity_reduction_vs_mtd=None,
            median_resistance_dominance_day=(
                float(np.median(dom_days)) if dom_days else None
            ),
            pfs_values=pfs_vals,
        )
        pfs_by_arm[name] = med
        tox_by_arm[name] = float(np.median(tox_vals))

    mtd_name = MTDPolicy().describe()
    if mtd_name in pfs_by_arm and pfs_by_arm[mtd_name] > 0:
        for name, res in results.items():
            res.toxicity_reduction_vs_mtd = 1.0 - res.median_toxicity / tox_by_arm[mtd_name]
        ratio = pfs_by_arm.get(AdaptiveDosePolicy().describe(), float("nan")) / pfs_by_arm[mtd_name]
    else:
        ratio = float("nan")

    notes = [
        "PFS is a simulation endpoint (RANO-fraction burden increase), not a "
        "clinical measurement; the calibrated dynamics class is anchored to "
        "the mCRPC adaptive-therapy trial, not to GBM outcomes",
        f"published anchor: adaptive/standard TTP ratio {ADAPTIVE_TTP_RATIO_ANCHOR:.2f} "
        f"({ADAPTIVE_TTP_MONTHS_MCRPC[0]} vs {ADAPTIVE_TTP_MONTHS_MCRPC[1]} months, mCRPC)",
        "this implementation reproduces the advantage's direction "
        f"(observed band {MEASURED_ADAPTIVE_MTD_RATIO_BAND[0]}-"
        f"{MEASURED_ADAPTIVE_MTD_RATIO_BAND[1]}x) but not the 2.34 magnitude; "
        "see CALIBRATION in the module docstring",
    ]
    verdict = (
        f"median PFS: " + ", ".join(f"{k}={v:.0f}d" for k, v in pfs_by_arm.items())
    )
    return TrialReport(
        arms=results,
        days=days,
        n_replicates=n_replicates,
        initial_resistant_fraction=initial_resistant_fraction,
        adaptive_vs_mtd_pfs_ratio=ratio,
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Threshold sweep + collateral-network contingency
# ---------------------------------------------------------------------------


@dataclass
class SweepReport:
    thetas: Tuple[float, ...]
    median_pfs: Tuple[float, ...]
    pfs_iqr: Tuple[Tuple[float, float], ...]
    best_theta: float
    best_band: Tuple[float, float]
    median_final_resistant_fraction: Tuple[float, ...] = ()
    no_collateral_median_pfs: Optional[Tuple[float, ...]] = None
    no_collateral_final_resistant_fraction: Optional[Tuple[float, ...]] = None
    notes: List[str] = field(default_factory=list)


def threshold_sweep(
    geometry: TumorGeometry,
    thetas: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60),
    *,
    theta_off_frac: float = 1 / 3,
    days: int = 300,
    n_replicates: int = 6,
    seed0: int = 100,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
    include_no_collateral: bool = True,
) -> SweepReport:
    """Sweep the switch-on threshold; optionally rerun with NO collateral
    sensitivity (Drug B kills R no better than S) to demonstrate that the
    switch policy's advantage is contingent on the collateral network, not
    guaranteed by the policy shape.

    The optimal threshold is reported as a BAND (all thetas within 5% of the
    best median PFS), because a point estimate from a simulation sweep is
    over-precision.
    """
    if len(thetas) < 2:
        raise ValueError("sweep needs at least two thresholds")
    sim = TumorEvolutionSimulator(geometry, clones=clones, drugs=drugs)

    def _run_sweep(drug_params: DrugParams) -> Tuple[List[float], List[Tuple[float, float]], List[float]]:
        sim2 = TumorEvolutionSimulator(geometry, clones=clones, drugs=drug_params)
        meds: List[float] = []
        iqrs: List[Tuple[float, float]] = []
        rfracs: List[float] = []
        for th in thetas:
            pol = CollateralSwitchPolicy(theta_on=th, theta_off=th * theta_off_frac)
            trajs = [
                sim2.run(
                    pol,
                    days=days,
                    initial_resistant_fraction=0.01,
                    seed=seed0 + rep,
                )
                for rep in range(n_replicates)
            ]
            vals = [t.pfs_value(days) for t in trajs]
            arr = np.asarray(vals)
            meds.append(float(np.median(arr)))
            iqrs.append(
                (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
            )
            rfracs.append(float(np.median([t.final_resistant_fraction for t in trajs])))
        return meds, iqrs, rfracs

    meds, iqrs, rfracs = _run_sweep(drugs)
    best = int(np.argmax(meds))
    best_val = meds[best]
    band = [thetas[i] for i, m in enumerate(meds) if m >= best_val * 0.95]

    no_coll: Optional[Tuple[float, ...]] = None
    no_coll_rfracs: Optional[Tuple[float, ...]] = None
    if include_no_collateral:
        # Kill R with B no better than S: the "no collateral sensitivity"
        # scenario. If switching still won here, the model would be rigged.
        nc = DrugParams(
            kill_max_S=drugs.kill_max_S,
            kill_max_R=(drugs.kill_max_R[0], drugs.kill_max_S[1]),
            ec50=drugs.ec50,
            diffusion=drugs.diffusion,
            clearance=drugs.clearance,
            toxicity_weight=drugs.toxicity_weight,
        )
        nc_meds, _, nc_rfracs = _run_sweep(nc)
        no_coll = tuple(nc_meds)
        no_coll_rfracs = tuple(nc_rfracs)

    notes = [
        "optimal threshold is reported as a band (within 5% of the best "
        "median PFS), not a point: a single-number optimum from a simulation "
        "sweep is over-precision",
    ]
    if no_coll is not None:
        delta = max(meds) - max(no_coll)
        if delta > 0:
            notes.append(
                f"no-collateral scenario: best median PFS {max(no_coll):.0f}d vs "
                f"{best_val:.0f}d with collateral sensitivity (delta {delta:+.0f}d) -- "
                "the switch policy's advantage is contingent on the drug-pair network"
            )
        else:
            # PFS censored in both networks: burden control is a property of
            # the policy shape, not the network. The network's effect shows
            # in whether the resistant clone is ever cleared.
            wrf = max(rfracs)
            orf = max(no_coll_rfracs) if no_coll_rfracs else float("nan")
            notes.append(
                "no-collateral scenario: median PFS indistinguishable (both "
                "censored) -- burden control comes from the policy shape, not "
                f"the network. The network's effect is on the resistant clone: "
                f"final resistant fraction {wrf:.2f} with collateral sensitivity "
                f"vs {orf:.2f} without. If switching cannot clear R, the "
                "schedule manages the tumour but never regains drug susceptibility"
            )
    return SweepReport(
        thetas=tuple(thetas),
        median_pfs=tuple(meds),
        pfs_iqr=tuple(iqrs),
        best_theta=thetas[best],
        best_band=(min(band), max(band)),
        median_final_resistant_fraction=tuple(rfracs),
        no_collateral_median_pfs=no_coll,
        no_collateral_final_resistant_fraction=no_coll_rfracs,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Spatial analytics: sanctuary sequestration
# ---------------------------------------------------------------------------


def sanctuary_sequestration(traj: SimulationTrajectory, geometry: TumorGeometry) -> float:
    """Resistant-clone enrichment in low-drug ('sanctuary') voxels.

    Mean resistant FRACTION (N_R/(N_S+N_R)) over the bottom-tercile Drug-A
    exposure voxels minus the same over the top-tercile. Positive values
    mean the resistant clone is concentrated where the drug is weakest --
    the sanctuary mechanism behind MTD failure.

    A raw density correlation is deliberately NOT used: once R dominates,
    its density tracks perfusion-driven GROWTH territory (high drug, killed
    S, open room), not refuge -- a density correlation then reads negative
    and would invert the mechanism it claims to measure. The R-fraction
    contrast measures where the clone is RELATIVELY enriched, which is the
    refuge question. Measure mid-trial (before global R dominance).
    """
    if traj.final_conc_A is None or traj.final_N_R is None or traj.final_N_S is None:
        raise ValueError("run the simulation with keep_fields=True for spatial analytics")
    mask = geometry.tumor_mask
    n_r = traj.final_N_R[mask]
    n_s = traj.final_N_S[mask]
    c_a = traj.final_conc_A[mask]
    total = n_s + n_r
    frac = np.divide(n_r, total, out=np.zeros_like(n_r), where=total > 0)
    lo, hi = np.quantile(c_a, [1.0 / 3.0, 2.0 / 3.0])
    sanctuary = frac[c_a <= lo]
    exposed = frac[c_a >= hi]
    if sanctuary.size == 0 or exposed.size == 0:
        return 0.0
    return float(sanctuary.mean() - exposed.mean())


# ---------------------------------------------------------------------------
# Provenance and honest limits
# ---------------------------------------------------------------------------


def evosteer_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "adaptive_vs_standard_ttp_ratio_mcrpc": EvidenceStatus.ESTABLISHED,
        "adaptive_therapy_failure_regime_low_resistant_fraction": EvidenceStatus.ESTABLISHED,
        "failure_regime_reproduction_in_spatial_abm": EvidenceStatus.PROVEN_FAILED,  # not reproduced here; see CALIBRATION (c)
        "rano_progression_threshold": EvidenceStatus.ESTABLISHED,
        "collateral_sensitivity_phenomenon": EvidenceStatus.ESTABLISHED,
        "collateral_sensitivity_network_specificity": EvidenceStatus.ESTABLISHED,  # contingent, pair-specific
        "logistic_growth_kill_diffusion_model_class": EvidenceStatus.ESTABLISHED,
        "gbm_growth_and_kill_constants": EvidenceStatus.UNKNOWN,  # bench-measurable (tier 3)
        "drug_diffusion_and_clearance_constants": EvidenceStatus.UNKNOWN,
        "pfs_endpoint_as_burden_fraction": EvidenceStatus.UNKNOWN,
    }


def what_this_model_cannot_know() -> List[str]:
    return [
        "patient-specific geometry: it consumes a segmentation mask; the "
        "phantom is a stand-in, not an MRI reconstruction",
        "immune clearance, angiogenesis dynamics, and vascular remodelling "
        "under therapy -- all real determinants of GBM drug delivery",
        "a third double-resistant clone: sequential therapy can select for "
        "double resistance, which two clones cannot represent",
        "GBM-specific parameter values: every rate is calibrated for the "
        "dynamics class and must be re-measured on the bench (see "
        "danon.validation tier 3) before any quantitative GBM claim",
        "clinical translation of the PFS endpoint: 'burden +25%' is a proxy "
        "for RANO progression, not RANO itself (no imaging, no non-enhancing "
        "component, no steroids)",
    ]


def print_evosteer_disclaimer() -> None:
    print(
        "=" * 72 + "\n"
        "EVOSTEER -- simulation-only evolutionary scheduling engine\n"
        "Anchors: mCRPC adaptive-therapy trial (dynamics class) + RANO\n"
        "progression fraction (endpoint). GBM parameters are ASSUMED and\n"
        "bench-measurable. No patient data, no clinical claim.\n"
        + "=" * 72
    )
