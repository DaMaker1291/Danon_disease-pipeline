"""Tissue portability: what is generic in the solver, and what is glioma-only.

A pitch claims two different things in one breath:

1. The *equations* are tissue-agnostic. A reaction-diffusion PDE, logistic
   growth and an S/R selection term do not know what organ they sit in. That
   is structurally true in this repository and it is MEASURABLE: the solver's
   entire tissue/drug surface is five dataclasses (``TumorGeometry``,
   ``CloneParams``, ``DrugParams``, ``ImmuneParams``, ``DiffusionField``),
   and the unmodified engine runs a deliberately non-brain phantom geometry
   deterministically and bounded, at 60 and 365 days, with no code change.

2. Therefore, the pitch continues, this software "generalizes to predict
   spatial recurrence and evolutionary resistance in metastatic breast, lung,
   and pancreatic cancers" by "swapping 90% of the codebase" against a
   four-row parameter table. That does NOT follow, and it is REFUSED here --
   with every mocked figure carried beside its refusal, the repository's
   standing convention.

WHAT IS MEASURED HERE
  * The swap surface itself: the five dataclasses and their field names,
    introspected live, so "what would have to be swapped" is a fact, not a
    percentage.
  * A portability run: ``generic_solid_mass`` -- a lobulated ellipsoid with
    uniform proliferation and no necrotic-core structure, i.e. NOT the GBM
    phantom -- through the real ``TumorEvolutionSimulator.run`` under MTD.
    Determinism, finiteness and the capacity bound are recorded and re-run
    live by the claim check.
  * The inventory that bounds the claim: the tissue diffusion table holds
    brain classes only (and is documented ASSUMED, with no DTI/perfusion
    series in this tree), the clone/drug/immune defaults are documented
    CALIBRATED or ASSUMED for GBM, and a live scan of ``data/`` file paths
    finds zero breast, lung or pancreatic assets and zero non-glioma drug
    molecule names.

WHAT IS REFUSED (mocked figure carried beside each refusal)
  * The pitch's three parameter-swap table rows (mammary / pulmonary /
    pancreatic meshes with ultrasound, PET and EUS streams and paclitaxel,
    osimertinib, gemcitabine).
  * "keep 90% of the codebase identical" -- a number that was never measured;
    what IS measured is the 23-field surface above, every value glioma-derived.
  * The judge script's generalization sentence (breast, lung, pancreas
    recurrence prediction).
  * The judge script's "built precisely to model those invisible residual
    cells after surgery" -- no post-operative imaging exists anywhere in this
    tree (BraTS-2021 is pre-operative only), and stream 1 is REFUSED on its
    live count of 0 cached captures (claim #49), so the post-surgical residual
    state is not an input this repository has ever had.
  * "Every cancer type benefits from pulsed or modulated drug scheduling" --
    schedule effects are measured only under this repository's
    glioma-calibrated parameters; the universal quantifier is unmeasured.

A successful run on other geometry is a property of the *code*, not evidence
about other cancers. The phantom in this module is not any organ's tissue --
no non-glioma tissue geometry exists in this repository to build one from --
and calling it "lung" would itself be mocked.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------


PORTABILITY_RECORD = "outputs_real/tumor_agnostic.json"

PORTABILITY_DISCLAIMER = (
    "A portability run measures that the unmodified solver accepts an "
    "arbitrary geometry and parameter set deterministically and bounded; "
    "the phantom is not any organ's tissue, and nothing here supports a "
    "prediction about any cancer other than glioblastoma."
)

#: File-path tokens scanned under ``data/``. Four groups: the three organ
#: families the pitch's table promises, plus the drug molecules it names.
#: File NAMES and relative paths only -- contents are out of scope, so a
#: refusal string in this repository can never trip the scan.
DATA_ORGAN_TOKENS: Dict[str, str] = {
    "breast": r"breast|mammary",
    "lung": r"lung|pulmonary",
    "pancreas": r"pancrea",
    "non_gbm_drugs": (
        r"paclitaxel|trastuzumab|osimertinib|pembrolizumab|gemcitabine|folfirinox"
    ),
}

BRAIN_TISSUE_CLASSES = ("edema", "grey_matter", "necrotic_core", "white_matter")

DAYS_SHORT = 60
DAYS_LONG = 365
SEED = 17
PHANTOM_SIZE = 24


# ---------------------------------------------------------------------------
# The non-brain phantom
# ---------------------------------------------------------------------------


def generic_solid_mass(size: int = PHANTOM_SIZE, seed: int = 0):
    """A lobulated solid mass that is deliberately NOT the GBM phantom.

    Differences from ``make_gbm_phantom`` are the point: ellipsoidal (aspect
    ~1.6 : 1.0 : 0.9) instead of spherical, uniform proliferation inside the
    mask instead of a hypoxic/necrotic core, no tissue-class structure at all.
    The geometry satisfies the ``TumorGeometry`` contract and nothing else
    about it is organ-specific.

    It is NOT "lung tissue" or "breast tissue": no such tissue geometry, image
    or parameter set exists in this repository to build one from, and this
    module says so rather than dressing a sphere up as an organ.
    """
    from danon.evosteer import TumorGeometry

    if size < 8:
        raise ValueError("phantom too small; use size >= 8")
    rng = np.random.default_rng(seed)
    idx = np.arange(size)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    c = (size - 1) / 2.0
    r = np.sqrt(((X - c) / 1.6) ** 2 + ((Y - c) / 1.0) ** 2 + ((Z - c) / 0.9) ** 2)
    radius = 0.40 * size
    wobble = 0.10 * radius * np.sin(1.3 * X + 0.4 * Z) * np.cos(1.1 * Y)
    tumor_mask = (r + wobble) < radius

    # Uniform proliferation: this phantom has no necrotic core by construction.
    proliferative = tumor_mask.astype(float)

    # A simple vascular bed so the contract's "no vasculature" guard passes:
    # one chord guaranteed through the centre, plus random chords.
    vessels = np.zeros((size, size, size), dtype=float)
    k = int(round(c))
    for axis in range(3):
        index: List[Any] = [k, k, k]
        index[axis] = slice(None)
        vessels[tuple(index)] = 1.0
    for _ in range(4):
        p0 = c + rng.normal(0, radius * 0.30, 3)
        direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        t_line = np.linspace(-radius * 1.4, radius * 1.4, 200)
        pts = p0[None, :] + t_line[:, None] * direction[None, :]
        for pt in pts:
            xi, yi, zi = (int(round(v)) for v in pt)
            if 0 <= xi < size and 0 <= yi < size and 0 <= zi < size:
                vessels[xi, yi, zi] = 1.0

    return TumorGeometry(
        tumor_mask=tumor_mask,
        proliferative=proliferative,
        vessels=vessels,
        voxel_mm=1.0,
    )


# ---------------------------------------------------------------------------
# The portability run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortabilityRun:
    days: int
    seed: int
    initial_burden: float
    final_burden: float
    max_daily_burden: float
    capacity: float
    capacity_ratio: float
    pfs_day: Optional[int]
    progressed: bool
    finite: bool
    bounded: bool


#: The structural ceiling this model implies, per voxel, from its two room
#: terms alone: S grows while N_S + 0.6*N_R < K (asymmetric competition, so S
#: crowds against only 0.6 of an R at cap K -> S reaches 0.4K -> total 1.4K),
#: and R grows only while total < K. Growth noise and conservative diffusion
#: in transit account for the small excess over 1.4 measured on this phantom;
#: 2x capacity is the hard bound those room terms imply, and exceeding it
#: would be a solver bug rather than a tumour.
BOUND_RATIO = 2.0


def portability_run(
    days: int, *, seed: int = SEED, size: int = PHANTOM_SIZE
) -> PortabilityRun:
    """Run the unmodified engine on the non-brain phantom under MTD.

    No clinical intake, no 5-ALA, no BraTS mask, no brain tissue class: the
    point under test is that ``TumorEvolutionSimulator.run`` needs none of
    them. Determinism, finiteness and the capacity bound are measured, not
    assumed.
    """
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator

    geom = generic_solid_mass(size=size, seed=0)
    sim = TumorEvolutionSimulator(geom)
    policy = MTDPolicy()
    traj = sim.run(policy, days=days, seed=seed)

    burden = np.asarray(traj.burden_S, dtype=float) + np.asarray(
        traj.burden_R, dtype=float
    )
    capacity = float(
        sim.clones.carrying_capacity * int(geom.tumor_mask.sum())
    )
    finite = bool(np.isfinite(burden).all())
    # See BOUND_RATIO: the model's growth equilibrium is 1.4x K per voxel
    # (asymmetric competition), so K*V is NOT the ceiling; 2x is.
    ratio = float(burden.max() / capacity) if capacity > 0 else float("inf")
    bounded = bool(ratio <= BOUND_RATIO)
    return PortabilityRun(
        days=int(days),
        seed=int(seed),
        initial_burden=float(burden[0]),
        final_burden=float(burden[-1]),
        max_daily_burden=float(burden.max()),
        capacity=capacity,
        capacity_ratio=ratio,
        pfs_day=traj.pfs_day,
        progressed=bool(traj.progressed),
        finite=finite,
        bounded=bounded,
    )


def portability_runs(
    seeds: Tuple[int, ...] = (SEED,), days: Tuple[int, ...] = (DAYS_SHORT, DAYS_LONG)
) -> List[PortabilityRun]:
    return [portability_run(d, seed=s) for s in seeds for d in days]


# ---------------------------------------------------------------------------
# The swap surface and the glioma-only inventory
# ---------------------------------------------------------------------------


def swap_surface() -> Dict[str, List[str]]:
    """The five dataclasses that are the solver's entire tissue/drug surface.

    Introspected live so the pitch's "swap the tissue and drug parameters"
    has a countable answer: these fields, and nothing else in the run path,
    is what a different tissue would have to supply.
    """
    from danon.anisotropy import DiffusionField
    from danon.evosteer import CloneParams, DrugParams, ImmuneParams, TumorGeometry

    return {
        cls.__name__: [f.name for f in dataclasses.fields(cls)]
        for cls in (TumorGeometry, CloneParams, DrugParams, ImmuneParams, DiffusionField)
    }


def swap_surface_fields() -> int:
    return sum(len(v) for v in swap_surface().values())


def tissue_inventory() -> Dict[str, Any]:
    """What fills the surface, checked where a live check is possible.

    * ``tissue_d_keys``: the diffusion literature table is brain classes only,
      and its own docstring says the values are ASSUMED, not measured (no DTI
      or perfusion series is present in this tree).
    * ``calibration_markers``: the provenance words carried by the defaults
      (CALIBRATED / ASSUMED), which is the repository admitting on the record
      that these values are glioma-calibrated assumptions.
    * ``data_organ_hits``: file-path scan of ``data/`` for the organs and
      molecules the pitch's table promises. Expected: nothing.
    """
    from danon.anisotropy import TISSUE_DRUG_D_MM2_PER_DAY
    from danon.evosteer import DEFAULT_CLONES, DEFAULT_DRUGS

    evosteer_src = ""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("evosteer.py",):
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                evosteer_src = fh.read()
            break

    markers = {
        "clone_params_calibrated": "CALIBRATED" in evosteer_src,
        "immune_params_assumed": "ASSUMED" in evosteer_src,
        "tissue_d_assumed": False,
    }
    try:
        import danon.anisotropy as _aniso

        aniso_src = ""
        path = _aniso.__file__
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                aniso_src = fh.read()
        markers["tissue_d_assumed"] = "ASSUMED here, not" in aniso_src or (
            "ASSUMED" in aniso_src and "TISSUE_DRUG_D_MM2_PER_DAY" in aniso_src
        )
    except Exception:
        pass

    return {
        "tissue_d_keys": sorted(TISSUE_DRUG_D_MM2_PER_DAY),
        "tissue_d_brain_only": sorted(TISSUE_DRUG_D_MM2_PER_DAY)
        == sorted(BRAIN_TISSUE_CLASSES),
        "default_drugs_diffusion": [float(v) for v in DEFAULT_DRUGS.diffusion],
        "default_clones_growth_S": float(DEFAULT_CLONES.growth_S),
        "calibration_markers": markers,
        "data_organ_hits": data_organ_hits(),
    }


def data_organ_hits(root: str = "data") -> Dict[str, List[str]]:
    """Scan file/dir paths under ``data/`` for the pitch's organs and drugs.

    File names and relative paths only (never contents), so this scan cannot
    be tripped by the refusals themselves.
    """
    import re

    out: Dict[str, List[str]] = {group: [] for group in DATA_ORGAN_TOKENS}
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames) + list(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace("\\", "/")
            for group, pattern in DATA_ORGAN_TOKENS.items():
                if re.search(pattern, rel, flags=re.IGNORECASE):
                    out[group].append(rel)
    return out


# ---------------------------------------------------------------------------
# Refusals: the mocked figures, carried beside the refusal
# ---------------------------------------------------------------------------


def portability_refusals() -> List[Dict[str, str]]:
    """Every cross-cancer claim in the pitch, refused with its mock held up."""
    return [
        {
            "refusal": "swap_table_breast",
            "pitch_says": (
                "Metastatic Breast Cancer | Mammary / Fatty Parenchyma Mesh | "
                "Automated Ultrasound + DCE-MRI | Paclitaxel / Trastuzumab"
            ),
            "refused_because": (
                "0 mammary meshes, 0 ultrasound/DCE-MRI assets and 0 paclitaxel "
                "or trastuzumab parameter sets exist in this tree (live file-path "
                "scan of data/), and the tissue diffusion table holds brain "
                "classes only."
            ),
        },
        {
            "refusal": "swap_table_lung",
            "pitch_says": (
                "Non-Small Cell Lung Cancer | Porous Pulmonary Alveolar Mesh | "
                "Low-Dose CT + PET Perfusion | Osimertinib / Pembrolizumab"
            ),
            "refused_because": (
                "0 pulmonary meshes, 0 CT/PET assets and 0 osimertinib or "
                "pembrolizumab parameter sets exist in this tree (live file-path "
                "scan of data/)."
            ),
        },
        {
            "refusal": "swap_table_pancreas",
            "pitch_says": (
                "Pancreatic Adenocarcinoma | Dense Fibrotic Stromal Mesh | "
                "Endoscopic Ultrasound + CT | Gemcitabine / FOLFIRINOX"
            ),
            "refused_because": (
                "0 pancreatic meshes, 0 EUS assets and 0 gemcitabine or "
                "FOLFIRINOX parameter sets exist in this tree (live file-path "
                "scan of data/)."
            ),
        },
        {
            "refusal": "ninety_percent_swap",
            "pitch_says": (
                "you keep 90% of the codebase identical and simply swap out "
                "the tissue and drug parameters"
            ),
            "refused_because": (
                "The 90% was never measured. What IS measured: the solver's "
                "whole tissue/drug surface is 5 dataclasses and "
                f"{swap_surface_fields()} fields (introspected live), and 100% "
                "of their values in this repository are glioma-derived or "
                "explicitly assumed."
            ),
        },
        {
            "refusal": "judge_sentence_generalizes",
            "pitch_says": (
                "this exact software framework generalizes to predict spatial "
                "recurrence and evolutionary resistance in metastatic breast, "
                "lung, and pancreatic cancers"
            ),
            "refused_because": (
                "Portability is a property of the code: the phantom run shows "
                "the unmodified solver executes on arbitrary geometry, but no "
                "non-glioma image, mesh, drug parameter set or outcome exists "
                "here to calibrate on or test against, so there is no measured "
                "breast, lung or pancreatic prediction to report."
            ),
        },
        {
            "refusal": "judge_sentence_postop_residual",
            "pitch_says": (
                "EvoTwin-10X was built precisely to model those invisible "
                "residual cells after surgery"
            ),
            "refused_because": (
                "No post-operative imaging exists anywhere in this tree (the "
                "cached BraTS-2021 cases are pre-operative only), so the "
                "post-surgical residual state is not an input this repository "
                "has ever had: the clinical intake plants on the pre-operative "
                "rim, and stream 1 is REFUSED on its live count of 0 cached "
                "captures (claim #49)."
            ),
        },
        {
            "refusal": "every_cancer_benefits_from_pulsing",
            "pitch_says": (
                "Every cancer type benefits from pulsed or modulated drug "
                "scheduling to delay the tipping point of resistant takeover."
            ),
            "refused_because": (
                "Schedule effects in this repository are measured only under "
                "the glioma-calibrated clone/drug/immune parameters, on "
                "phantoms and committed BraTS-2021 cases. 'Every cancer type' "
                "is a universal quantifier over diseases that have no data, "
                "mesh or parameter set here."
            ),
        },
    ]


def portability_provenance() -> Dict[str, EvidenceStatus]:
    """The cross-cancer claims, on the record as PROVEN_FAILED."""
    return {
        "predicts_breast_lung_pancreas_recurrence": EvidenceStatus.PROVEN_FAILED,
        "non_brain_mesh_exists_here": EvidenceStatus.PROVEN_FAILED,
        "non_brain_drug_parameter_set_exists": EvidenceStatus.PROVEN_FAILED,
        "postoperative_residual_input_available_here": EvidenceStatus.PROVEN_FAILED,
    }


def what_portability_cannot_show() -> List[str]:
    return [
        "The phantom is not lung, breast or pancreas tissue -- no such tissue "
        "geometry, image or parameter set exists in this repository to build "
        "one from; calling it an organ would be mocked.",
        "A run that executes on other geometry says the CODE is generic; it "
        "says nothing about whether any equation describes another cancer.",
        "Every value on the swap surface is glioma-derived or explicitly "
        "assumed (CALIBRATED / ASSUMED markers are recorded live).",
        "The tissue diffusion table holds brain classes only, and its own "
        "docstring marks the values ASSUMED -- no DTI or perfusion series is "
        "present in this tree.",
        "No non-glioma outcome exists here to test a prediction against, so "
        "cross-cancer accuracy has no denominator.",
    ]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def tumor_agnostic_report() -> Dict[str, Any]:
    runs = portability_runs()
    geom = generic_solid_mass()
    inventory = tissue_inventory()
    return {
        "disclaimer": PORTABILITY_DISCLAIMER,
        "swap_surface": swap_surface(),
        "swap_surface_fields": swap_surface_fields(),
        "phantom": {
            "size": PHANTOM_SIZE,
            "shape": [int(v) for v in geom.tumor_mask.shape],
            "tumour_voxels": int(geom.tumor_mask.sum()),
            "proliferation": "uniform (no necrotic-core structure)",
            "note": (
                "deliberately not make_gbm_phantom and not any organ's tissue; "
                "it exists to measure that the solver needs no brain input"
            ),
        },
        "runs": [asdict(r) for r in runs],
        "bound_note": (
            "capacity = K x mask voxels. The model's growth equilibrium is "
            "1.4x capacity (S crowds against only 0.6 of an R at cap), so "
            "bounded means <= 2.0x capacity (BOUND_RATIO); the measured max "
            "on this phantom is recorded per run, not clipped."
        ),
        "inventory": inventory,
        "refusals": portability_refusals(),
        "provenance": {
            k: str(v.value) for k, v in portability_provenance().items()
        },
        "notes": what_portability_cannot_show(),
    }
