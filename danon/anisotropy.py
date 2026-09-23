"""
Anisotropic, heterogeneous transport: the unit bridge and the real operator.

WHY THIS MODULE EXISTS
----------------------
Two upgrades are proposed for the spatial engine: replace the scalar drug
diffusion constant with a spatially varying tensor `div(D(x) grad C)`, and give
the cell-invasion term a named motility coefficient. Both are implemented here,
and implementing them surfaced a defect that dominates either one.

THE UNIT TRAP
-------------
The engine measures diffusion in **voxel^2/day**, not mm^2/day. On the synthetic
phantom `TumorGeometry(voxel_mm=1.0)`, so `diffusion=0.35` reads as
0.35 mm^2/day -- which happens to sit inside the literature range for white
matter. On real geometry the documented decimation of a 1 mm scan by (4, 4, 3)
gives a voxel edge of ~3.67 mm, so **the same 0.35 is 4.71 mm^2/day**: 13.5x the
value it appears to be, with `voxel_mm` marked "bookkeeping only" so nothing
raises. A parameter calibrated on the phantom therefore changes its physical
meaning by 13.47x when moved to a real case, silently.

Measured, at the two voxel sizes the repository actually uses:

    quantity                     phantom (1.00 mm)      real (3.67 mm)
    drug D  = 0.35 voxel^2/day     0.35 mm^2/day         4.71 mm^2/day
    cell D  = 0.02 voxel^2/day     0.02 mm^2/day         0.27 mm^2/day

Against the tissue values this project is being asked to model -- white-matter
fibre 0.35, edema 0.25, necrotic core 0.01 mm^2/day, and cell motility
0.001-0.005 mm^2/day -- the engine's cell term is 4-20x too fast on the phantom
and 54-270x too fast on real geometry. `unit_audit()` reports this rather than
asserting a direction of error.

THE OPERATOR, AND WHY IT IS NOT `D * laplacian(C)`
--------------------------------------------------
When `D` varies in space, `div(D grad C)` and `D * laplacian(C)` are different
operators -- they differ by `grad(D) . grad(C)`, which is exactly the term that
makes a necrotic core a barrier rather than just a slow region. The correct
discretisation is finite-volume: flux across each voxel face uses the
**harmonic** mean of the two adjacent `D` values, because conductances in series
add inversely. `divergence_d_grad` implements that, and its identity on constant
`D` is pinned by a test against the engine's own `_laplacian`, so "same
operator in the constant case" is verified rather than assumed.

Anisotropy is carried as a per-voxel diagonal tensor (D_xx, D_yy, D_zz). That is
the part that matters physically -- directional diffusion along fibre tracts
versus across them -- and unlike a full 3x3 tensor it is testable and cannot
introduce a non-symmetric, energy-creating stencil.

NOT CLAIMED
-----------
* The tissue `D` values are literature-derived, not measured on this cohort.
* No DTI or perfusion dataset is present, so the tissue-class map is synthetic
  unless a caller supplies a real one.
* Making transport heterogeneous does not make the engine's clinical output
  valid; it changes a declared assumption into a spatially varying one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------
# The unit bridge
# ---------------------------------------------------------------------------

#: Voxel edge of the engine's synthetic phantom, where the constants were tuned.
PHANTOM_VOXEL_MM = 1.0

#: Voxel edge of the real decimated grid. A 1 mm scan decimated by (4, 4, 3)
#: gives (4, 4, 3) mm spacings, and the engine's single pitch is their mean --
#: the same convention `danon.realdata` uses (`voxel_mm = np.mean(source_vox)`) and
#: the one `danon.claims.check_voxel_scale` verifies. An earlier draft of this
#: module used the geometric mean instead, which would have been a third value
#: for a quantity the repository already fixes.
REAL_DECIMATION = (4, 4, 3)
REAL_SOURCE_VOXEL_MM = 1.0
REAL_VOXEL_MM = float(
    sum(REAL_SOURCE_VOXEL_MM * f for f in REAL_DECIMATION) / len(REAL_DECIMATION)
)


def voxel_area_mm2(voxel_mm: float) -> float:
    """Area of one voxel face in mm^2. The conversion factor is its square."""
    if voxel_mm <= 0:
        raise ValueError("voxel_mm must be positive")
    return float(voxel_mm) ** 2


def d_to_engine(d_mm2_per_day: float, voxel_mm: float) -> float:
    """mm^2/day -> voxel^2/day, the unit the engine integrates in."""
    return float(d_mm2_per_day) / voxel_area_mm2(voxel_mm)


def d_to_physical(d_voxel2_per_day: float, voxel_mm: float) -> float:
    """voxel^2/day -> mm^2/day. The inverse, used by the audit."""
    return float(d_voxel2_per_day) * voxel_area_mm2(voxel_mm)


# ---------------------------------------------------------------------------
# Tissue diffusion coefficients
# ---------------------------------------------------------------------------

#: Literature-derived tissue diffusion coefficients, in **mm^2/day**. The values
#: are the ones this project is being asked to model; they are ASSUMED here, not
#: measured, because no DTI or perfusion series is present in the repository.
TISSUE_DRUG_D_MM2_PER_DAY: Dict[str, float] = {
    "white_matter": 0.35,
    "edema": 0.25,
    "necrotic_core": 0.01,
    "grey_matter": 0.20,
}

#: Cell motility (Fisher-Kolmogorov `D_cell`) in mm^2/day. Two orders of
#: magnitude below drug diffusion: cells crawl, small molecules diffuse.
CELL_MOTILITY_MM2_PER_DAY_RANGE: Tuple[float, float] = (0.001, 0.005)

#: The engine's hardcoded cell-migration coefficient, in voxel^2/day. It is a
#: literal `0.02` inside `TumorEvolutionSimulator.run`, so it carries no unit
#: and no provenance; this constant names it so the audit can price it.
LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY = 0.02


@dataclass(frozen=True)
class UnitAuditRow:
    """One quantity, at both voxel sizes, with its physical value."""

    name: str
    value_voxel2_per_day: float
    phantom_mm2_per_day: float
    real_mm2_per_day: float
    reference_range_mm2_per_day: Optional[Tuple[float, float]]
    note: str

    @property
    def inflation_factor(self) -> float:
        return voxel_area_mm2(REAL_VOXEL_MM) / voxel_area_mm2(PHANTOM_VOXEL_MM)

    @property
    def within_reference(self) -> Optional[bool]:
        if self.reference_range_mm2_per_day is None:
            return None
        lo, hi = self.reference_range_mm2_per_day
        return lo <= self.real_mm2_per_day <= hi

    def describe(self) -> str:
        ref = (
            f"reference [{self.reference_range_mm2_per_day[0]}, "
            f"{self.reference_range_mm2_per_day[1]}] mm^2/day"
            if self.reference_range_mm2_per_day
            else "no reference range"
        )
        verdict = (
            "within" if self.within_reference else "OUTSIDE"
        ) if self.within_reference is not None else "n/a"
        return (
            f"{self.name}: {self.value_voxel2_per_day:g} voxel^2/day = "
            f"{self.phantom_mm2_per_day:.4g} mm^2/day on the phantom "
            f"({PHANTOM_VOXEL_MM:g} mm voxel) but "
            f"{self.real_mm2_per_day:.4g} mm^2/day on real geometry "
            f"({REAL_VOXEL_MM:.2f} mm voxel); {ref}; {verdict}. {self.note}"
        )


def unit_audit() -> List[UnitAuditRow]:
    """Every diffusion constant the engine carries, priced in physical units."""
    wm = TISSUE_DRUG_D_MM2_PER_DAY["white_matter"]
    return [
        UnitAuditRow(
            name="drug diffusion (DrugParams.diffusion)",
            value_voxel2_per_day=0.35,
            phantom_mm2_per_day=d_to_physical(0.35, PHANTOM_VOXEL_MM),
            real_mm2_per_day=d_to_physical(0.35, REAL_VOXEL_MM),
            reference_range_mm2_per_day=(wm, wm),
            note=(
                "on the phantom this coincidentally equals the white-matter "
                "value, which is why the mismatch is invisible until a real "
                "grid is used"
            ),
        ),
        UnitAuditRow(
            name="cell motility (hardcoded 0.02 in run())",
            value_voxel2_per_day=LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY,
            phantom_mm2_per_day=d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, PHANTOM_VOXEL_MM),
            real_mm2_per_day=d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, REAL_VOXEL_MM),
            reference_range_mm2_per_day=CELL_MOTILITY_MM2_PER_DAY_RANGE,
            note=(
                "even on the phantom this sits above the literature motility "
                "range, and `voxel_mm` is documented as bookkeeping only, so "
                "nothing raises when it moves to a real grid"
            ),
        ),
    ]


def unit_audit_report() -> str:
    lines = [
        "DIFFUSION UNITS: the engine's constants are voxel^2/day, not mm^2/day",
        f"  phantom voxel {PHANTOM_VOXEL_MM:g} mm; real decimation "
        f"{REAL_DECIMATION} of a {REAL_SOURCE_VOXEL_MM:g} mm scan -> "
        f"{REAL_VOXEL_MM:.3f} mm edge",
        f"  the same number is "
        f"{voxel_area_mm2(REAL_VOXEL_MM) / voxel_area_mm2(PHANTOM_VOXEL_MM):.2f}x "
        f"larger in physical units on real geometry than on the phantom",
        "",
    ]
    lines += [f"  {row.describe()}" for row in unit_audit()]
    lines += [
        "",
        "  This is not a modelling choice that can be left implicit: a value",
        "  tuned on the phantom changes its physical meaning by that factor the",
        "  moment it is used on a real case. `d_to_engine` / `d_to_physical` are",
        "  the bridge, and `DiffusionField.from_tissue_classes` applies it.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The variable-coefficient operators
# ---------------------------------------------------------------------------


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Face conductance from two voxel conductances in series.

    Zero on either side gives zero: a face touching an impermeable voxel
    carries no flux. The arithmetic mean would instead let a necrotic core leak
    half of its neighbour's diffusion, which is the error this function exists
    to avoid.
    """
    out = np.zeros_like(a, dtype=float)
    good = (a > 0.0) & (b > 0.0)
    denom = a[good] + b[good]
    out[good] = 2.0 * a[good] * b[good] / denom
    return out


def divergence_d_grad(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """`div(D(x) grad C)` on a uniform grid with zero-flux boundaries.

    Finite-volume, second order, with harmonic face conductances. On constant
    `D` this reduces **exactly** to `D * laplacian(C)` (pinned by a test against
    `danon.evosteer._laplacian`), which is what makes it a drop-in for the
    scalar operator rather than a different model.
    """
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    if c.shape != d.shape or c.ndim != 3:
        raise ValueError("c and d must be matching 3D arrays")
    if np.any(d < 0.0):
        raise ValueError("diffusion coefficients must be non-negative")

    cp = np.pad(c, 1, mode="edge")
    dp = np.pad(d, 1, mode="edge")
    centre = (slice(1, -1), slice(1, -1), slice(1, -1))
    c_centre = cp[centre]
    d_centre = dp[centre]

    out = np.zeros_like(c_centre)
    for axis in range(3):
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[axis] = slice(0, -2)
        hi[axis] = slice(2, None)
        lo = tuple(lo)
        hi = tuple(hi)
        # Edge-padded C equals the centre value at the boundary, so the boundary
        # face gradient is exactly zero -- zero-flux, without a special case.
        f_lo = _harmonic_mean(dp[lo], d_centre)
        f_hi = _harmonic_mean(d_centre, dp[hi])
        out += f_hi * (cp[hi] - c_centre) - f_lo * (c_centre - cp[lo])
    return out


def divergence_diag_tensor_grad(c: np.ndarray, d_diag: np.ndarray) -> np.ndarray:
    """`div(diag(Dxx, Dyy, Dzz)(x) grad C)` for a per-voxel diagonal tensor.

    Anisotropy enters as directional fibre diffusion: `Dxx != Dyy != Dzz`. With
    `Dxx == Dyy == Dzz` this equals `divergence_d_grad`, and with a constant
    diagonal it equals `D * laplacian(C)` -- both pinned by tests.
    """
    c = np.asarray(c, dtype=float)
    d_diag = np.asarray(d_diag, dtype=float)
    if d_diag.shape != (3,) + c.shape:
        raise ValueError("d_diag must have shape (3,) + c.shape")
    if np.any(d_diag < 0.0):
        raise ValueError("diffusion coefficients must be non-negative")

    cp = np.pad(c, 1, mode="edge")
    centre = (slice(1, -1), slice(1, -1), slice(1, -1))
    c_centre = cp[centre]
    out = np.zeros_like(c_centre)
    for axis in range(3):
        dp = np.pad(d_diag[axis], 1, mode="edge")
        d_centre = dp[centre]
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[axis] = slice(0, -2)
        hi[axis] = slice(2, None)
        f_lo = _harmonic_mean(dp[tuple(lo)], d_centre)
        f_hi = _harmonic_mean(d_centre, dp[tuple(hi)])
        out += f_hi * (cp[tuple(hi)] - c_centre) - f_lo * (
            c_centre - cp[tuple(lo)]
        )
    return out


# ---------------------------------------------------------------------------
# Tissue-class diffusion fields
# ---------------------------------------------------------------------------

#: Canonical integer labels for a tissue-class map.
TISSUE_LABELS: Dict[int, str] = {
    0: "necrotic_core",
    1: "edema",
    2: "white_matter",
    3: "grey_matter",
}


@dataclass(frozen=True)
class DiffusionField:
    """Per-voxel drug diffusion in **engine units**, built from tissue classes.

    `d_engine` is what the simulator integrates with; `d_mm2_per_day` is kept so
    the physical values remain inspectable after conversion, which is the whole
    point of having the bridge.
    """

    d_engine: np.ndarray
    d_mm2_per_day: np.ndarray
    voxel_mm: float
    provenance: Dict[str, EvidenceStatus] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.d_engine.shape

    @property
    def contrast_ratio(self) -> float:
        """Max/min ratio across voxels: how much the field varies."""
        lo = float(self.d_engine.min())
        hi = float(self.d_engine.max())
        if lo <= 0:
            return float("inf")
        return hi / lo

    def describe(self) -> str:
        return (
            f"diffusion field {self.shape} at {self.voxel_mm:.2f} mm/voxel: "
            f"{self.d_mm2_per_day.min():.4g}-{self.d_mm2_per_day.max():.4g} "
            f"mm^2/day (contrast {self.contrast_ratio:.0f}x)"
        )

    @classmethod
    def from_tissue_classes(
        cls,
        classes: np.ndarray,
        *,
        voxel_mm: float = REAL_VOXEL_MM,
        d_mm2_per_day: Optional[Dict[str, float]] = None,
        anisotropy: Optional[np.ndarray] = None,
    ) -> "DiffusionField":
        """Build a field from integer tissue labels.

        `anisotropy`, when given, is a per-voxel directional factor applied to
        the x axis only, so a fibre tract can run one way and not another. It is
        applied before the unit conversion, so it is dimensionless and cannot
        smuggle a unit error in.
        """
        table = dict(d_mm2_per_day or TISSUE_DRUG_D_MM2_PER_DAY)
        classes = np.asarray(classes)
        if classes.ndim != 3:
            raise ValueError("classes must be a 3D label array")
        unknown = sorted(set(np.unique(classes).tolist()) - set(TISSUE_LABELS))
        if unknown:
            raise ValueError(
                f"unknown tissue label(s) {unknown}; labels are {sorted(TISSUE_LABELS)}"
            )
        physical = np.zeros(classes.shape, dtype=float)
        for label, name in TISSUE_LABELS.items():
            physical[classes == label] = table[name]
        if anisotropy is not None:
            aniso = np.asarray(anisotropy, dtype=float)
            if aniso.shape != classes.shape:
                raise ValueError("anisotropy must match classes")
            if np.any(aniso < 0.0):
                raise ValueError("anisotropy factors must be non-negative")
            physical = physical * aniso
        engine = physical / voxel_area_mm2(voxel_mm)
        return cls(
            d_engine=engine,
            d_mm2_per_day=physical,
            voxel_mm=voxel_mm,
            provenance={
                "tissue_d_values": EvidenceStatus.UNKNOWN,
                "unit_conversion": EvidenceStatus.ESTABLISHED,
                "tissue_class_map": EvidenceStatus.UNKNOWN,
            },
        )

    @classmethod
    def uniform(cls, value_mm2_per_day: float, shape: Sequence[int], *, voxel_mm: float = REAL_VOXEL_MM):
        physical = np.full(tuple(shape), float(value_mm2_per_day))
        return cls(
            d_engine=physical / voxel_area_mm2(voxel_mm),
            d_mm2_per_day=physical,
            voxel_mm=voxel_mm,
            provenance={"tissue_d_values": EvidenceStatus.UNKNOWN,
                        "unit_conversion": EvidenceStatus.ESTABLISHED,
                        "tissue_class_map": EvidenceStatus.ESTABLISHED},
        )


# ---------------------------------------------------------------------------
# What this changes, measured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportContrast:
    """Scalar vs heterogeneous transport on the same geometry, same drug."""

    n_voxels: int
    uniform_d_mm2_per_day: float
    core_d_mm2_per_day: float
    fibre_d_mm2_per_day: float
    max_ratio: float
    operator_gap_at_core: float

    def describe(self) -> str:
        return (
            f"{self.n_voxels} voxels: uniform {self.uniform_d_mm2_per_day:.4g} "
            f"mm^2/day vs core {self.core_d_mm2_per_day:.4g} and fibre "
            f"{self.fibre_d_mm2_per_day:.4g} -> {self.max_ratio:.0f}x contrast; "
            f"`D*laplacian` and `div(D grad)` differ by "
            f"{self.operator_gap_at_core:.1%} of the divergence at the core face"
        )


def measure_transport_contrast(
    classes: np.ndarray,
    concentration: np.ndarray,
    *,
    voxel_mm: float = REAL_VOXEL_MM,
) -> TransportContrast:
    """How far apart the scalar and heterogeneous operators actually are.

    Both operators are evaluated on the same field. The gap is reported as a
    fraction of the heterogeneous divergence's own magnitude, so it answers "how
    wrong would the scalar approximation be here" rather than "are they equal".
    """
    classes = np.asarray(classes)
    concentration = np.asarray(concentration, dtype=float)
    field = DiffusionField.from_tissue_classes(classes, voxel_mm=voxel_mm)
    uniform_mm2 = TISSUE_DRUG_D_MM2_PER_DAY["white_matter"]
    uniform = np.full(classes.shape, d_to_engine(uniform_mm2, voxel_mm))

    het = divergence_d_grad(concentration, field.d_engine)
    scalar = uniform * _engine_laplacian(concentration)
    denom = float(np.abs(het).max())
    gap = float(np.abs(het - scalar).max() / denom) if denom > 0 else float("nan")
    core = TISSUE_DRUG_D_MM2_PER_DAY["necrotic_core"]
    fibre = TISSUE_DRUG_D_MM2_PER_DAY["white_matter"]
    return TransportContrast(
        n_voxels=int(classes.size),
        uniform_d_mm2_per_day=uniform_mm2,
        core_d_mm2_per_day=core,
        fibre_d_mm2_per_day=fibre,
        max_ratio=fibre / core,
        operator_gap_at_core=gap,
    )


def _engine_laplacian(f: np.ndarray) -> np.ndarray:
    """The engine's own zero-flux 3D Laplacian (see `danon.evosteer`)."""
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


def tissue_classes_from_geometry(geometry: object) -> np.ndarray:
    """Derive a tissue-class map from a `TumorGeometry` without new data.

    A real case replaces this with a segmentation model's output. Deriving it
    here from the geometry the engine already has means the heterogeneous path
    can be exercised on real BraTS-derived cases without pretending a
    segmentation exists: necrotic core is the low-proliferation interior, edema
    the tumour rim, white matter everything else.
    """
    prolif = np.asarray(geometry.proliferative, dtype=float)
    mask = np.asarray(geometry.tumor_mask, dtype=bool)
    classes = np.full(mask.shape, TISSUE_LABELS_INVERSE["white_matter"], dtype=int)
    inside = mask & (prolif < 0.25)
    rim = mask & ~inside
    classes[inside] = TISSUE_LABELS_INVERSE["necrotic_core"]
    classes[rim] = TISSUE_LABELS_INVERSE["edema"]
    return classes


#: Reverse lookup, so the mapping is declared once.
TISSUE_LABELS_INVERSE: Dict[str, int] = {v: k for k, v in TISSUE_LABELS.items()}


def anisotropy_provenance() -> Dict[str, EvidenceStatus]:
    """What each quantity here is, in the repo's provenance vocabulary."""
    return {
        "unit_conversion": EvidenceStatus.ESTABLISHED,
        "harmonic_face_conductance": EvidenceStatus.ESTABLISHED,
        "constant_d_identity": EvidenceStatus.ESTABLISHED,
        "tissue_d_values": EvidenceStatus.UNKNOWN,
        "cell_motility_range": EvidenceStatus.UNKNOWN,
        "tissue_class_map": EvidenceStatus.UNKNOWN,
        "anisotropy_field": EvidenceStatus.UNKNOWN,
    }


def what_anisotropy_cannot_show() -> List[str]:
    return [
        "No DTI or perfusion series is present, so tissue classes are derived "
        "from proliferation, not measured by imaging.",
        "The tissue diffusion values are literature constants, not fitted to this "
        "cohort; changing them changes the model, not the biology.",
        "Fixing the unit bridge makes the transport physics consistent across grid "
        "sizes. It does not make the engine's clinical output valid.",
        "A heterogeneous field can only ever slow or redirect transport relative "
        "to a scalar field of the same mean, so any claim of improved delivery "
        "from this change alone would be backwards.",
    ]


__all__ = [
    "CELL_MOTILITY_MM2_PER_DAY_RANGE",
    "LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY",
    "PHANTOM_VOXEL_MM",
    "REAL_DECIMATION",
    "REAL_VOXEL_MM",
    "TISSUE_DRUG_D_MM2_PER_DAY",
    "TISSUE_LABELS",
    "TISSUE_LABELS_INVERSE",
    "DiffusionField",
    "TransportContrast",
    "UnitAuditRow",
    "anisotropy_provenance",
    "d_to_engine",
    "d_to_physical",
    "divergence_d_grad",
    "divergence_diag_tensor_grad",
    "measure_transport_contrast",
    "tissue_classes_from_geometry",
    "unit_audit",
    "unit_audit_report",
    "voxel_area_mm2",
    "what_anisotropy_cannot_show",
]
