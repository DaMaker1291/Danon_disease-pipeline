"""Longitudinal walk-forward evaluation: the harness, and what it refuses to score.

WHY THIS MODULE EXISTS
----------------------
"Feed the engine day 0, run it forward, compare with the patient's follow-up scan"
is the right test for a mechanistic model, and it is one function call away here:
`TumorEvolutionSimulator.run` already accepts an initial state, and the engine
exposes `field_hook`/`observation_hook`. What is missing is not code. It is the
PAIR -- a baseline segmentation and a follow-up segmentation of the same patient
in the same image space -- and there are none in this repository:

  * CFB-GBM (TCIA, DOI 10.7937/v9pn-2f72) ships an availability MANIFEST here:
    393 usable pairs over 264 patients, no images;
  * MU-Glioma-Post (TCIA, DOI 10.7937/7k9k-3c83) documents 203 subjects with 596
    post-treatment timepoints and segmentations, and the public NBIA API returns
    0 patients and 0 series for that collection from this machine, because the
    images are behind the licence login;
  * GLASS is a genomics consortium release under controlled access, and no GLASS
    record exists in this tree at all.

So this module does three things, and refuses a fourth.

1. THE METRICS, VERIFIED ON PLANTED GROUND TRUTH. 3D Dice, Jaccard, a symmetric
   95th-percentile surface distance (EDT-based), and the TRUE Jensen-Shannon
   divergence -- not scipy's `jensenshannon`, which returns its SQUARE ROOT.
   That distinction decides whether "JSD < 0.15" means 0.15 bits or 0.0225 bits,
   and a target quoted without a base is not a target.

2. THE NULL MODELS A PREDICTION MUST BEAT. A Dice number means nothing on its
   own: recurrence overlaps the treated field, so "the tumour never changed" is
   a strong competitor. Every prediction is therefore scored against persistence
   and against margin dilation at several radii, and the comparison is reported
   beside the score rather than under it.

3. THE STRUCTURAL BOUND. The engine has no migration and no invasion term: cell
   growth is multiplied by the per-voxel `proliferative` map, which is zero
   outside the baseline mask, so simulated burden can only DISAPPEAR from where
   it started. `expansion_bound` measures that on a real case, in the most
   favourable configuration available (proliferation set to 1.0 everywhere in
   the mask, drug inlet uniform, both arms). If nothing leaves the mask even
   then, no recurrence-geometry Dice can separate this engine from persistence:
   it can only score worse, and the deficit is shrinkage.

WHAT IS REFUSED, AND WHY IT IS NOT MODESTY
------------------------------------------
- A pair that is not in the same image space. Two segmentations from different
  acquisitions have different frames of reference; Dice between them measures the
  registration error, not the model. This repository has no co-registration, so
  it refuses rather than silently scoring an alignment it never performed.
- An empty follow-up segmentation. An empty observation is not a null result, it
  is no observation.
- A Jensen-Shannon divergence between supports that are not the same. The engine
  carries exactly two bins (drug-sensitive, drug-resistant). A GLASS allele-
  frequency spectrum over k mutations is a different support; comparing a 2-bin
  simulation to a k-bin spectrum would require summarising the genomics down to
  two bins by a rule this repository has no basis for. `clonal_jsd` checks the
  supports and refuses when they differ.
- Any headline number. There is no measured Dice and no measured JSD anywhere in
  this module, because there is no pair to measure them on.

WHAT A POSITIVE RESULT WOULD TAKE
---------------------------------
A registered longitudinal pair (baseline plus a follow-up at a stated interval),
the treatment actually given between them, and an invasion or migration term in
the engine -- in that order. The first two are data acquisition; the third is the
one that makes a spatial forecast falsifiable, and `expansion_bound` measures how
far this repository is from it today.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    TumorEvolutionSimulator,
    TumorGeometry,
)
from danon.uncertainty import ARM_POLICIES

#: Recurrence appears at the margin of the treated field, so the competitor a
#: spatial forecast must beat is the baseline dilated by a few millimetres --
#: not the empty set. These radii are the nulls scored here.
MARGIN_DILATION_MM: Tuple[float, ...] = (2.0, 5.0, 10.0)

#: A prediction is scored as a set of voxels holding live cells. The engine's own
#: MIN_VIABLE_DENSITY already empties voxels below half a cell, so "greater than
#: zero" is the engine's own definition of presence, not a threshold invented
#: here.
PRESENCE_THRESHOLD = 0.0

#: The census of longitudinal sources this repository can see, filled by
#: `_walkforward_census.py` (the network part is recorded there, with its query
#: and its date, because a census that cannot be re-run offline must still be
#: auditable).
CENSUS_PATH = os.path.join("outputs_real", "walkforward_census.json")


class WalkForwardError(ValueError):
    """Raised instead of scoring something that cannot be scored."""


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _as_bool(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise WalkForwardError(f"{name} must be a 3D mask, got shape {a.shape}")
    return a > 0


def _check_same_grid(predicted: np.ndarray, observed: np.ndarray) -> None:
    if predicted.shape != observed.shape:
        raise WalkForwardError(
            f"predicted {predicted.shape} and observed {observed.shape} are "
            "different grids; two segmentations can only be compared in one "
            "frame of reference"
        )


def dice3d(predicted: np.ndarray, observed: np.ndarray) -> float:
    """3D Dice similarity. Two EMPTY masks are refused, not scored as 1.0.

    Returning 1.0 for a prediction that contains nothing would reward a model
    for erasing the tumour, which is exactly the failure mode a shrinkage-only
    engine can reach.
    """
    p = _as_bool(predicted, "predicted")
    o = _as_bool(observed, "observed")
    _check_same_grid(p, o)
    denom = int(p.sum()) + int(o.sum())
    if denom == 0:
        raise WalkForwardError(
            "both masks are empty: Dice is undefined for two absent tumours, "
            "and 1.0 would score erasure as a perfect prediction"
        )
    return 2.0 * float(np.logical_and(p, o).sum()) / denom


def jaccard3d(predicted: np.ndarray, observed: np.ndarray) -> float:
    """3D Jaccard (intersection over union), the volumetric analogue of Dice."""
    p = _as_bool(predicted, "predicted")
    o = _as_bool(observed, "observed")
    _check_same_grid(p, o)
    union = int(np.logical_or(p, o).sum())
    if union == 0:
        raise WalkForwardError("both masks are empty: Jaccard is undefined")
    return float(np.logical_and(p, o).sum()) / union


def surface_mask(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: those with at least one face neighbour outside the mask.

    Voxels on the array border count as surface (their outside neighbour is not
    tumour). The same convention `danon.grid_artifact._boundary_voxels` counts.
    """
    m = _as_bool(mask, "mask")
    padded = np.pad(m, 1, constant_values=False)
    interior = np.ones_like(m, dtype=bool)
    for axis in range(3):
        for delta in (-1, 1):
            sl = [slice(1, -1)] * 3
            sl[axis] = slice(1 + delta, padded.shape[axis] - 1 + delta)
            interior &= padded[tuple(sl)]
    return np.logical_and(m, ~interior)


def surface_distance_p95(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    voxel_mm: float = 1.0,
) -> float:
    """Symmetric 95th-percentile surface distance in mm.

    Both directions are pooled (predicted surface to observed, observed surface
    to predicted), so under- and over-segmentation both count. Euclidean
    distance transform on the voxel grid, scaled by `voxel_mm`; an empty
    prediction has no surface, so it is refused rather than returned as inf.
    """
    if voxel_mm <= 0.0:
        raise WalkForwardError(f"voxel_mm must be positive, got {voxel_mm}")
    p = _as_bool(predicted, "predicted")
    o = _as_bool(observed, "observed")
    _check_same_grid(p, o)
    sp, so = surface_mask(p), surface_mask(o)
    if not sp.any() or not so.any():
        raise WalkForwardError(
            "a surface distance needs a surface on both sides; an empty "
            "prediction has none"
        )
    from scipy import ndimage

    # Distance from every voxel to the nearest OBSERVED surface voxel, read at
    # the predicted surface, and the reverse.
    d_to_obs = ndimage.distance_transform_edt(~so, sampling=float(voxel_mm))
    d_to_pred = ndimage.distance_transform_edt(~sp, sampling=float(voxel_mm))
    pooled = np.concatenate([d_to_obs[sp], d_to_pred[so]])
    return float(np.percentile(pooled, 95.0))


def shannon_entropy(p: np.ndarray, *, base: float = 2.0) -> float:
    """Shannon entropy in `base` units, with a hard zero convention."""
    q = np.asarray(p, dtype=float)
    if np.any(q < 0):
        raise WalkForwardError("a probability distribution cannot be negative")
    total = float(q.sum())
    if total <= 0:
        raise WalkForwardError("a distribution must carry mass")
    q = q / total
    nz = q[q > 0]
    return float(-(nz * (np.log(nz) / np.log(base))).sum())


def jensen_shannon_divergence(
    p: Sequence[float],
    q: Sequence[float],
    *,
    base: float = 2.0,
) -> float:
    """The TRUE Jensen-Shannon divergence in `base` units (bits by default).

        JSD(p, q) = H((p + q)/2) - (H(p) + H(q)) / 2

    NOT `scipy.spatial.distance.jensenshannon`, which returns the SQUARE ROOT of
    this quantity AND defaults to NATURAL units (base=None): for p = [0.95,
    0.05], q = [0.12, 0.88] this function returns 0.5886 bits while scipy
    returns 0.6387 = sqrt(0.5886 * ln 2). A target of "JSD < 0.15" is therefore
    0.15 bits here, 0.0225 bits = 0.0156 nats with scipy's squared, and a target
    quoted without a base is not a target. Both are pinned in the tests.
    """
    a = np.asarray(p, dtype=float)
    b = np.asarray(q, dtype=float)
    if a.shape != b.shape:
        raise WalkForwardError(
            f"JSD needs two distributions on the SAME support, got shapes "
            f"{a.shape} and {b.shape}"
        )
    if a.size < 2:
        raise WalkForwardError("JSD needs at least two bins")
    if np.any(a < 0) or np.any(b < 0):
        raise WalkForwardError("a probability distribution cannot be negative")
    if a.sum() <= 0 or b.sum() <= 0:
        raise WalkForwardError("a distribution must carry mass")
    m = 0.5 * (a / a.sum() + b / b.sum())
    return shannon_entropy(m, base=base) - 0.5 * (
        shannon_entropy(a, base=base) + shannon_entropy(b, base=base)
    )


def clonal_jsd(
    simulated_sensitive: float,
    simulated_resistant: float,
    observed: Sequence[float],
    *,
    base: float = 2.0,
) -> float:
    """JSD between the engine's own two-bin clone state and an observation.

    The engine carries exactly two bins, so an observation with any other number
    of bins is a DIFFERENT SUPPORT and is refused. A k-bin allele-frequency
    spectrum cannot be scored against a 2-bin simulation without a summarisation
    rule, and this repository has none -- inventing one would put a number on the
    poster that measures the summarisation rather than the model.
    """
    expected = np.asarray([simulated_sensitive, simulated_resistant], dtype=float)
    if expected.min() < 0:
        raise WalkForwardError("clone fractions cannot be negative")
    obs = np.asarray(observed, dtype=float)
    if obs.size != expected.size:
        raise WalkForwardError(
            f"the engine has {expected.size} clone bins but the observation has "
            f"{obs.size}: JSD between different supports is not a divergence, it "
            "is a category error"
        )
    return jensen_shannon_divergence(expected, obs, base=base)


# ---------------------------------------------------------------------------
# The pair, and the nulls it defines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LongitudinalPair:
    """One patient's baseline and follow-up segmentation, with its provenance.

    `registered_same_space` is a DECLARATION, not a measurement: this repository
    has no co-registration machinery, so a pair from different acquisitions must
    be registered by the person supplying it, and the flag records that they did.
    """

    patient_id: str
    baseline_mask: np.ndarray
    followup_mask: np.ndarray
    interval_days: int
    voxel_mm: float = 1.0
    source: str = "unspecified"
    registered_same_space: bool = False

    def __post_init__(self) -> None:
        b = _as_bool(self.baseline_mask, "baseline_mask")
        f = _as_bool(self.followup_mask, "followup_mask")
        if b.shape != f.shape:
            raise WalkForwardError(
                f"{self.patient_id}: baseline {b.shape} and follow-up {f.shape} "
                "are different grids; register them to one frame of reference "
                "before scoring anything"
            )
        if not self.registered_same_space:
            raise WalkForwardError(
                f"{self.patient_id}: the pair is not declared registered. Dice "
                "between two unregistered segmentations measures the alignment "
                "error, not the model"
            )
        if int(self.interval_days) <= 0:
            raise WalkForwardError(
                f"{self.patient_id}: interval_days must be positive, got "
                f"{self.interval_days}"
            )
        if float(self.voxel_mm) <= 0.0:
            raise WalkForwardError(
                f"{self.patient_id}: voxel_mm must be positive, got {self.voxel_mm}"
            )
        if int(b.sum()) == 0:
            raise WalkForwardError(
                f"{self.patient_id}: the baseline segmentation is empty, so there "
                "is no day 0 to run forward from"
            )
        if int(f.sum()) == 0:
            raise WalkForwardError(
                f"{self.patient_id}: the follow-up segmentation is empty. An "
                "empty observation is not a null result, it is no observation"
            )

    @property
    def baseline_voxel_count(self) -> int:
        return int((self.baseline_mask > 0).sum())

    @property
    def followup_voxel_count(self) -> int:
        return int((self.followup_mask > 0).sum())

    def volume_ratio(self) -> float:
        """Observed follow-up volume / baseline volume, in the same voxel units."""
        return self.followup_voxel_count / self.baseline_voxel_count

    def describe(self) -> str:
        return (
            f"{self.patient_id}: {self.baseline_voxel_count:,} -> "
            f"{self.followup_voxel_count:,} voxels over {self.interval_days} d "
            f"({self.source})"
        )


def dilate_mask(mask: np.ndarray, radius_mm: float, *, voxel_mm: float = 1.0) -> np.ndarray:
    """Morphological dilation by a Euclidean radius, the margin-recurrence null."""
    if radius_mm < 0:
        raise WalkForwardError(f"radius must be non-negative, got {radius_mm}")
    m = _as_bool(mask, "mask")
    if radius_mm == 0:
        return m.copy()
    from scipy import ndimage

    d = ndimage.distance_transform_edt(~m, sampling=float(voxel_mm))
    return d <= float(radius_mm)


def matched_volume_sphere(
    mask: np.ndarray,
    extra_voxels: int,
    *,
    voxel_mm: float = 1.0,
) -> np.ndarray:
    """The isotropic-growth null: `extra_voxels` grown evenly in all directions.

    "The tumour became a ball of the same size" is the baseline a spatial
    forecast has to beat, and it has to be built at MATCHED VOLUME or the
    comparison measures size rather than shape. Voxels outside the mask are
    ranked by Euclidean distance to it and the nearest `extra_voxels` are added,
    which is the discrete isotropic expansion.
    """
    m = _as_bool(mask, "mask")
    if int(extra_voxels) < 0:
        raise WalkForwardError(f"extra_voxels must be non-negative, got {extra_voxels}")
    if int(extra_voxels) == 0:
        return m.copy()
    from scipy import ndimage

    d = ndimage.distance_transform_edt(~m, sampling=float(voxel_mm))
    outside = np.flatnonzero(~m.ravel())
    if outside.size == 0:
        raise WalkForwardError(
            "there is nothing outside the mask to grow into: the tumour fills the grid"
        )
    take = min(int(extra_voxels), outside.size)
    order = outside[np.argsort(d.ravel()[outside], kind="stable")[:take]]
    grown = m.copy().ravel()
    grown[order] = True
    return grown.reshape(m.shape)


def shape_departure(
    predicted: np.ndarray,
    baseline: np.ndarray,
    *,
    voxel_mm: float = 1.0,
) -> Dict[str, Any]:
    """How far a predicted margin is from growing evenly in all directions.

    Returns the Dice between the prediction and the matched-volume isotropic
    null, so 1.0 means the forecast is a ball and a smaller number means the
    forecast has structure. It REFUSES the anisotropy reading when the
    prediction adds no voxels at all: a prediction that only shrinks has no
    margin to be anisotropic, and scoring its overlap with the baseline as
    'ball-like' would dress shrinkage up as shape.
    """
    pred = _as_bool(predicted, "predicted")
    base = _as_bool(baseline, "baseline")
    _check_same_grid(pred, base)
    extra_pred = int(np.logical_and(pred, ~base).sum())
    lost = int(np.logical_and(base, ~pred).sum())
    if extra_pred == 0:
        raise WalkForwardError(
            "the prediction adds nothing outside the day-0 mask, so it has no "
            "margin whose shape could be measured; what happened here is "
            f"shrinkage of {lost} voxels, not an isotropic forecast"
        )
    sphere = matched_volume_sphere(base, extra_pred, voxel_mm=voxel_mm)
    return {
        "predicted_voxels_outside_baseline": extra_pred,
        "predicted_voxels_lost_from_baseline": lost,
        "dice_vs_matched_volume_sphere": dice3d(pred, sphere),
        "jaccard_vs_matched_volume_sphere": jaccard3d(pred, sphere),
        "sphere_voxels_outside_baseline": extra_pred,
        "reading": (
            "1.0 would mean the predicted margin is an isotropic ball of the "
            "same size; the distance below 1 is the shape the transport field "
            "contributed, and it is NOT evidence that the shape is correct"
        ),
    }


def null_models(pair: LongitudinalPair, *, radii_mm: Sequence[float] = MARGIN_DILATION_MM) -> Dict[str, Any]:
    """The competitors a spatial forecast has to beat, scored on the same pair.

    persistence      the day-0 mask, unchanged -- wins wherever recurrence is
                     confined to the treated volume, which is common;
    margin_<r>_mm    the day-0 mask dilated by r mm, the clinical expectation
                     that recurrence appears at the margin;
    relocation       the same voxel count placed uniformly at random, which is
                     an uninformative floor rather than a real competitor, and is
                     reported as such.
    """
    nulls: Dict[str, Any] = {}
    variants: Dict[str, np.ndarray] = {"persistence": pair.baseline_mask > 0}
    for r in radii_mm:
        variants[f"margin_{r:g}mm"] = dilate_mask(
            pair.baseline_mask, float(r), voxel_mm=pair.voxel_mm
        )
    rng = np.random.default_rng(0)
    n = pair.baseline_voxel_count
    flat = np.zeros(pair.baseline_mask.size, dtype=bool)
    flat[rng.choice(flat.size, size=min(n, flat.size), replace=False)] = True
    variants["relocation"] = flat.reshape(pair.baseline_mask.shape)

    for name, mask in variants.items():
        entry: Dict[str, Any] = {
            "voxel_count": int(mask.sum()),
            "dice": dice3d(mask, pair.followup_mask),
            "jaccard": jaccard3d(mask, pair.followup_mask),
        }
        try:
            entry["surface_distance_p95_mm"] = surface_distance_p95(
                mask, pair.followup_mask, voxel_mm=pair.voxel_mm
            )
        except WalkForwardError:
            entry["surface_distance_p95_mm"] = None
        nulls[name] = entry
    return nulls


def score_prediction(predicted: np.ndarray, pair: LongitudinalPair) -> Dict[str, Any]:
    """Score one predicted follow-up mask against the observed one and the nulls."""
    pred = _as_bool(predicted, "predicted")
    _check_same_grid(pred, pair.followup_mask)
    entry: Dict[str, Any] = {
        "patient_id": pair.patient_id,
        "predicted_voxel_count": int(pred.sum()),
        "observed_voxel_count": pair.followup_voxel_count,
        "dice": dice3d(pred, pair.followup_mask),
        "jaccard": jaccard3d(pred, pair.followup_mask),
        "surface_distance_p95_mm": surface_distance_p95(
            pred, pair.followup_mask, voxel_mm=pair.voxel_mm
        ),
        # Growth outside the baseline mask: the quantity a mechanism must get
        # right and a shrinkage-only engine cannot produce at all.
        "predicted_voxels_outside_baseline": int(
            np.logical_and(pred, ~(pair.baseline_mask > 0)).sum()
        ),
        "observed_voxels_outside_baseline": int(
            np.logical_and(pair.followup_mask > 0, ~(pair.baseline_mask > 0)).sum()
        ),
    }
    nulls = null_models(pair)
    # The matched-volume isotropic null is scored on the OBSERVED growth, so the
    # baseline is the same size as the recurrence the prediction is compared to.
    extra_observed = int(
        np.logical_and(pair.followup_mask > 0, ~(pair.baseline_mask > 0)).sum()
    )
    if extra_observed > 0:
        sphere = matched_volume_sphere(
            pair.baseline_mask, extra_observed, voxel_mm=pair.voxel_mm
        )
        entry["matched_volume_sphere_null"] = {
            "dice": dice3d(sphere, pair.followup_mask),
            "jaccard": jaccard3d(sphere, pair.followup_mask),
            "extra_voxels": extra_observed,
        }
        nulls["matched_volume_sphere"] = entry["matched_volume_sphere_null"]
    entry["nulls"] = nulls
    entry["beats_persistence"] = bool(entry["dice"] > nulls["persistence"]["dice"])
    entry["best_null"] = max(nulls, key=lambda k: nulls[k]["dice"])
    entry["margin_over_best_null"] = entry["dice"] - nulls[entry["best_null"]]["dice"]
    return entry


# ---------------------------------------------------------------------------
# The structural bound: what this engine can and cannot predict about space
# ---------------------------------------------------------------------------


def migration_audit(voxel_mm: float = 1.0) -> Dict[str, Any]:
    """The cell-motility coefficient this engine expands with, in physical units.

    The run loop adds `motility * _laplacian(N)` each day, with `motility`
    defaulting to `danon.anisotropy.LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY` = 0.02
    **voxel**^2/day. Because `TumorGeometry.voxel_mm` is documented as
    bookkeeping only, that literal is a different physical diffusivity on every
    grid: `d_to_physical` converts it, and the arithmetic is reported here rather
    than restated in prose.
    """
    from danon.anisotropy import (
        CELL_MOTILITY_MM2_PER_DAY_RANGE,
        LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY,
        d_to_physical,
    )

    physical = d_to_physical(float(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY), float(voxel_mm))
    lo, hi = CELL_MOTILITY_MM2_PER_DAY_RANGE
    return {
        "value_voxel2_per_day": float(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY),
        "voxel_mm": float(voxel_mm),
        "physical_mm2_per_day": float(physical),
        "reference_range_mm2_per_day": [float(lo), float(hi)],
        "multiple_of_range_top": float(physical / hi) if hi else None,
        "inside_reference_range": bool(lo <= physical <= hi),
    }


def expansion_bound(
    baseline_mask: np.ndarray,
    *,
    arm: str = "mtd",
    days: int = 365,
    seeds: Sequence[int] = (101,),
    r0: float = 0.01,
    fill_fraction: float = 0.65,
    voxel_mm: float = 1.0,
    proliferative_inside: float = 1.0,
    cell_motility: Optional[float] = None,
) -> Dict[str, Any]:
    """Where the engine's predicted tumour can be, and what puts it there.

    Growth in this engine is `r * N * room * proliferative * noise`, so with a
    proliferative map that is zero outside the baseline mask, mass outside the
    mask is never GROWN -- the only thing that can move cells outwards is the
    daily `motility * laplacian(N)` term. This measures both halves of that
    statement on the same seeds and the most favourable legal configuration for
    expansion (proliferation 1.0 everywhere inside, uniform drug inlet, the arm
    named):

      * with the engine's default motility: the share of final burden sitting
        outside the baseline mask;
      * with motility forced to 0: the same share, which must then be exactly 0,
        isolating the migration term as the sole source of any outward movement.

    The coefficient is reported in physical units by `migration_audit`, because
    "how wide is the predicted margin" is a question about that coefficient and
    not about the biology.
    """
    mask = _as_bool(baseline_mask, "baseline_mask")
    if int(mask.sum()) == 0:
        raise WalkForwardError("an empty baseline mask has nothing to expand")
    if arm not in ARM_POLICIES:
        raise WalkForwardError(f"arm must be one of {sorted(ARM_POLICIES)}, got {arm!r}")
    if not 0.0 <= float(proliferative_inside) <= 1.0:
        raise WalkForwardError(
            f"proliferative_inside must be in [0, 1], got {proliferative_inside}"
        )
    proliferative = np.where(mask, float(proliferative_inside), 0.0)
    geom = TumorGeometry(
        tumor_mask=mask,
        proliferative=proliferative,
        vessels=np.where(mask, 0.5, 0.0),
        voxel_mm=float(voxel_mm),
    )

    def run_config(motility: Optional[float]) -> List[Dict[str, Any]]:
        per_seed: List[Dict[str, Any]] = []
        for seed in seeds:
            sim = TumorEvolutionSimulator(
                geom,
                clones=DEFAULT_CLONES,
                drugs=DEFAULT_DRUGS,
                cell_motility=motility,
            )
            traj = sim.run(
                ARM_POLICIES[arm](),
                days=int(days),
                initial_resistant_fraction=float(r0),
                seed=int(seed),
                fill_fraction=float(fill_fraction),
                keep_fields=True,
            )
            if traj.final_N_S is None or traj.final_N_R is None:
                raise WalkForwardError(
                    "the run did not retain its final fields, so the expansion "
                    "bound cannot be measured; keep_fields=True is required"
                )
            total = np.asarray(traj.final_N_S) + np.asarray(traj.final_N_R)
            per_seed.append(
                {
                    "seed": int(seed),
                    "burden_total": float(total.sum()),
                    "burden_outside_baseline_mask": float(total[~mask].sum()),
                    "burden_outside_share": (
                        float(total[~mask].sum() / total.sum())
                        if total.sum() > 0
                        else 0.0
                    ),
                    "voxels_alive": int((total > PRESENCE_THRESHOLD).sum()),
                    "voxels_alive_outside": int(
                        (total[~mask] > PRESENCE_THRESHOLD).sum()
                    ),
                    "dice_predicted_vs_baseline": dice3d(
                        total > PRESENCE_THRESHOLD, mask
                    ),
                    "pfs_day": traj.pfs_day,
                }
            )
        return per_seed

    default_runs = run_config(cell_motility)
    frozen_runs = run_config(0.0)
    default_share = max(r["burden_outside_share"] for r in default_runs)
    frozen_share = max(r["burden_outside_share"] for r in frozen_runs)
    return {
        "arm": arm,
        "days": int(days),
        "seeds": [int(s) for s in seeds],
        "proliferative_inside": float(proliferative_inside),
        "voxel_mm": float(voxel_mm),
        "default_motility_runs": default_runs,
        "zero_motility_runs": frozen_runs,
        "max_burden_outside_baseline_share": default_share,
        "max_burden_outside_share_with_motility_zero": frozen_share,
        "expansion_requires_the_migration_term": bool(
            frozen_share == 0.0 and default_share > 0.0
        ),
        "migration": migration_audit(voxel_mm),
        # The structural half of the statement, reported as the number it is:
        # the growth-rate map summed over every voxel outside the mask.
        "growth_rate_map_sum_outside_mask": float(proliferative[~mask].sum()),
        "note": (
            "the per-voxel proliferative map has growth rate 0 outside the "
            "baseline mask, so outward movement is diffusion of cells and "
            "nothing else; with motility 0 the predicted tumour cannot leave "
            "the mask at all"
        ),
    }


# ---------------------------------------------------------------------------
# The census: what longitudinal pairs exist for this repository to score
# ---------------------------------------------------------------------------


def longitudinal_census(
    *,
    census_path: str = CENSUS_PATH,
    manifest_path: str = os.path.join(
        "data", "cfb_gbm", "CFB-GBM_mri_availability_v02_20260129.tsv"
    ),
) -> Dict[str, Any]:
    """Live, offline counts of what can be scored right now, plus the record.

    The repository-side counts are re-derived on every call: how many usable
    pairs the cached CFB-GBM manifest declares, whether any image or
    segmentation of it is present, and how many GLASS records this tree holds.
    The network-derived counts come from `_walkforward_census.py`'s record,
    because a census that needs the network cannot be re-run in a claim check.
    """
    out: Dict[str, Any] = {
        "scorable_pairs": 0,
        "pairs_with_cached_segmentations": 0,
        "sources": {},
        "manifest_path": manifest_path,
    }

    if os.path.exists(manifest_path):
        # Reuse the tested loader rather than re-parsing the long-format
        # manifest: its pair rule (same patient, t1gd present, >= 4 weeks apart)
        # is the one the rest of the repository quotes, and a second
        # implementation of it would drift.
        from danon.availability import mri_availability_census

        census = mri_availability_census(manifest_path)
        out["sources"]["CFB-GBM"] = {
            "manifest_rows": int(census.n_rows),
            "patients_in_manifest": int(census.n_patients),
            "timepoints_by_label": dict(census.n_timepoints_by_label),
            "usable_pairs_declared": int(census.n_usable_pairs),
            "patients_with_usable_pair": int(census.n_patients_with_usable_pair),
            "min_pair_interval_weeks": int(census.min_interval_weeks),
            "median_pair_interval_weeks": census.interval_weeks_median,
            "max_pair_interval_weeks": census.interval_weeks_max,
            "same_machine_pairs": int(census.n_same_machine_pairs),
            "cross_machine_pairs": int(census.n_cross_machine_pairs),
            "images_present": False,
            "note": (
                "availability manifest only: it declares which MRI timepoints "
                "exist on TCIA, and no image or segmentation from it is cached "
                "here, so nothing in it is scorable -- and a pair that crosses "
                "scanners is a delta-volume confound before it is a test case"
            ),
        }

    glass_hits: List[str] = []
    for root, _dirs, files in os.walk("."):
        if any(part in root for part in (".git", "node_modules", "__pycache__")):
            continue
        for name in files:
            if "glass" in name.lower():
                glass_hits.append(os.path.join(root, name))
    # GLASS is the one source a walk-forward route names whose data IS publicly
    # reachable, and the refusal was re-derived rather than inherited: the
    # processed release is an open cBioPortal study, its specimens are paired,
    # and its mutation records DO carry the allele counts a clonal frequency is
    # computed from. So the blocker is not access and not the frequency -- it is
    # the MAPPING from a per-variant allele fraction onto this engine's initial
    # resistant POPULATION fraction, which the engine has no space for.
    # What this tree now HOLDS for GLASS, read back from the artifacts rather
    # than promised: the channel module, the raw cache, and the measured record
    # the cohort script wrote. A sentence claiming these records are measured
    # here has to fail if they are not, so the check below reads this block.
    variant_channel: Dict[str, Any] = {
        "module": os.path.join("danon", "variants.py"),
        "raw_cache": os.path.join("data", "glass_paired", "glass_mutations.json"),
        "artifact": os.path.join("outputs_real", "glass_variants.json"),
        "built": False,
        "measured": None,
        "refusals_recorded": None,
        "error": None,
    }
    if os.path.exists(variant_channel["artifact"]):
        try:
            with open(variant_channel["artifact"], encoding="utf-8") as fh:
                built = json.load(fh)
            variant_channel["built"] = True
            variant_channel["measured"] = built.get("measured")
            variant_channel["refusals_recorded"] = sorted(
                (built.get("what_is_refused") or {}).keys()
            )
        except (OSError, ValueError) as exc:  # noqa: BLE001 -- recorded, not raised
            variant_channel["error"] = f"{type(exc).__name__}: {exc}"

    out["sources"]["GLASS"] = {
        "files_in_tree": len(glass_hits),
        "usable_pairs_declared": 0,
        "access": (
            "PUBLIC for the processed release: cBioPortal publishes the study "
            "'difg_glass' (Diffuse Glioma, GLASS Consortium) with publicStudy "
            "true, 693 samples of which 629 are sequenced, queryable through an "
            "open API with no application. Only the RAW resource goes through "
            "terms (Synapse glass; dbGaP phs001460), so 'controlled access' is "
            "the wrong reason to give for refusing this"
        ),
        "pairing": (
            "genuinely paired specimens of the same patient: SAMPLE_TYPE counts "
            "Tumor Primary 309, First Recurrence 318, Second Recurrence 49, "
            "Third Recurrence 14, Fourth recurrence 2, First Metastasis 1, with "
            "ids naming the pair (GLSS-AT-00P6-TP beside GLSS-AT-00P6-R1)"
        ),
        "frequencies": (
            "present, which is the part this repository previously got wrong: "
            "the mutation records carry tumorAltCount/tumorRefCount, so a VAF is "
            "computable -- e.g. MSH6 R922* in GLSS-CU-R007-R1 with alt 53 / ref "
            "84, VAF 0.387"
        ),
        "why_it_is_still_not_usable": (
            "PARTLY SUPERSEDED, and what remains refused is narrower than it "
            "was. The records are now IN this tree and are measured by "
            "danon.variants: a paired ALLELE-FRACTION spectrum per patient, and "
            "a provable LOWER bound on the fraction of tumour cells carrying a "
            "variant (f >= A_t needs no copy number) at both specimens of each "
            "primary -> recurrence pair. Still refused, and for the stated "
            "reason rather than for want of data: a POINT clone fraction and an "
            "UPPER bound both need a tumour purity and a local copy number this "
            "release does not carry, and the k-bin spectrum still cannot be "
            "scored as the engine's two-bin CELL-POPULATION fraction, because "
            "clonal_jsd refuses a support of a different size by construction"
        ),
        "variant_channel": variant_channel,
        "note": (
            f"{len(glass_hits)} GLASS record(s) exist in this tree: the raw "
            "mutation cache and the measured artifact named under "
            "'variant_channel'. Every number in the artifact is measured from "
            "the public release's own read counts, and every mapping past a "
            "lower bound is refused there rather than here"
        ),
    }

    # LUMIERE: reachable, longitudinal, and -- for the spatial-Dice test --
    # still not scorable, for a reason that is about the metric rather than
    # availability. The collection is on Figshare (NOT the TCIA/NBIA APIs the
    # probe above uses), and its two assets are separate: the segmentations are
    # AUTOMATED (two U-Net tools), and the EXPERT contribution is a per-timepoint
    # RANO response class. A Dice against automated masks is model-versus-model,
    # not model-versus-truth, so caching it would not make a Dice measurable.
    lumiere_hits: List[str] = []
    for root, _dirs, files in os.walk("."):
        if any(part in root for part in (".git", "node_modules", "__pycache__")):
            continue
        for name in files:
            if "lumiere" in name.lower():
                lumiere_hits.append(os.path.join(root, name))
    out["sources"]["LUMIERE"] = {
        "files_in_tree": len(lumiere_hits),
        "host": "Figshare (Springer Nature), collection 5904905 -- not TCIA/NBIA",
        "declared_by": (
            "Suter et al., Scientific Data 9:768 (2022), and the collection's "
            "own article list read back through the Figshare API"
        ),
        "patients": 91,
        "study_dates": 638,
        "images": 2487,
        "expert_rano_ratings": 616,
        "expert_asset": (
            "a per-timepoint RANO response class (progressive / stable / partial / "
            "complete), its rating rationale, and bi-dimensional Macdonald "
            "measurements -- a response label, NOT a segmentation"
        ),
        "segmentations": (
            "AUTOMATED: DeepBraTumIA and HD-GLIO-AUTO, two U-Net tools, shipped "
            "as the collection's own separate article titled 'MRI data and "
            "automated segmentations'"
        ),
        "pathology": (
            "MGMT promoter methylation (37 methylated / 43 unmethylated / 11 "
            "unavailable) and IDH1; NO Ki-67, so it cannot supply the growth-rate "
            "mapping either"
        ),
        "treatment": (
            "single-centre and single-regimen: resection plus temozolomide-based "
            "chemo-radiation (Inselspital Bern, 2008-2013) -- no adaptive arm and "
            "no dose variation, so it cannot test steering against MTD"
        ),
        "licence": (
            "ambiguous across sources: the paper is CC BY 4.0, the dataset's own "
            "repository states non-commercial use; unresolved here"
        ),
        "scorable_for_a_spatial_dice": False,
        "note": (
            "reachable and genuinely longitudinal, but it does not satisfy the "
            "spatial-Dice requirement: its masks are automated, so a Dice against "
            "them scores this engine against another algorithm, and its expert "
            "asset is a response class rather than a mask. Nothing from it is "
            "cached here, so the 0-scorable-pairs refusal is unchanged. What it "
            "could support is a simulated-versus-expert time-to-progression "
            "comparison -- a different endpoint, and not implemented."
        ),
    }

    recorded: Optional[Dict[str, Any]] = None
    if os.path.exists(census_path):
        # `json` is imported at module level; a local import here would make
        # `json` a function-local name and shadow the arifact read above it.
        with open(census_path, encoding="utf-8") as fh:
            recorded = json.load(fh)
        out["recorded_tcia"] = recorded.get("tcia", {})
        out["recorded_figshare"] = recorded.get("figshare", {})
        out["recorded_cbioportal"] = recorded.get("cbioportal", {})
        out["recorded_on"] = recorded.get("recorded_on")
    out["record_source"] = census_path if recorded else None
    out["scorable_pairs"] = out["pairs_with_cached_segmentations"]
    out["what_would_make_them_scorable"] = [
        "a baseline segmentation and a follow-up segmentation of the same patient "
        "in ONE frame of reference (co-registration is not implemented here)",
        "the interval in days between the two scans, and the treatment actually "
        "given between them",
        "for the clonal readout, a paired genomic measurement summarised onto the "
        "engine's two bins by a rule this repository would have to state and "
        "defend",
    ]
    return out


def walkforward_provenance() -> Dict[str, EvidenceStatus]:
    """What the walk-forward test has, and what it measurably does not.

    `expansion_needs_the_migration_term` is ESTABLISHED and `motility_is_in_
    physical_units` is PROVEN_FAILED: the outward term exists and is the only
    one, but it is a voxel-unit literal whose physical value depends on the
    grid, which `migration_audit` reports rather than assumes.
    """
    return {
        "dice_and_jaccard_algebra": EvidenceStatus.ESTABLISHED,
        "true_jensen_shannon_vs_scipy_sqrt": EvidenceStatus.ESTABLISHED,
        "surface_distance_edt": EvidenceStatus.ESTABLISHED,
        "expansion_needs_the_migration_term": EvidenceStatus.ESTABLISHED,
        "motility_is_in_physical_units": EvidenceStatus.PROVEN_FAILED,
        "engine_dice_against_a_patient_followup": EvidenceStatus.PROVEN_FAILED,
        "engine_jsd_against_glass": EvidenceStatus.PROVEN_FAILED,
        "co_registration": EvidenceStatus.PROVEN_FAILED,
        "invasion_term": EvidenceStatus.PROVEN_FAILED,
    }


def what_walk_forward_cannot_show() -> List[str]:
    return [
        "anything about a patient: there is no registered longitudinal pair in "
        "this repository, so no Dice and no JSD has been measured on one",
        "the registration it never performed: a Dice between two unregistered "
        "segmentations is an alignment error wearing a model metric's name",
        "the clonal composition of a tumour from a single segmentation: the "
        "engine's resistant fraction is a state variable, not a measured "
        "genotype, and no assay here reports one",
        "clinical validity: a Dice that beats persistence would show the front "
        "propagates like invasion does, not that the treatment effect is right",
    ]
