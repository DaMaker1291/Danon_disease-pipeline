"""
Real AAV9 capsid structure intake — RCSB PDB 3UX1.

WHY THIS MODULE EXISTS
----------------------
The capsid layer of this repository used to carry a hand-written table of
"CA coordinates from PDB 3J1S" for the VR-IV and VR-VIII loops. Three things
were wrong with it, and all three are the same failure mode — a fabricated
number wearing the costume of a measured one:

1. **Wrong serotype.** PDB 3J1S is *adeno-associated virus 2 in complex with
   the neutralising monoclonal antibody A20* (McCraw et al., J Virol
   2012;86(23):12722-12730). It is not AAV9. The AAV9 reference structure is
   **3UX1** (DiMattia et al., J Virol 2012;86(12):6947-6958).
2. **The coordinates were invented.** The table's values put the VR-IV loop
   ~38 A from the capsid centre. The real AAV9 VR-IV window sits at
   116.8-143.7 A, with its outermost atom *defining* the capsid surface
   radius. An invented coordinate set cannot be phase-shifted into agreement;
   it describes a different object.
3. **The citations were mixed.** The 3J1S paper's citation was attached to the
   AAV9 claim, so the error was self-consistent enough to survive review.

This module replaces the table with the deposited structure itself. Nothing
here is fitted, smoothed, or approximated:

* the C-alpha coordinates are read from the PDB file, byte for byte;
* the 60-subunit T=1 assembly is generated from the entry's own
  `REMARK 350 BIOMT` matrices — the depositors' symmetry operators, not
  ours;
* the 3-fold / 5-fold / 2-fold axes are *derived* by classifying those
  operators by rotation angle, so they are the capsid's real symmetry
  elements rather than idealised icosahedron directions;
* per-residue burial is a contact number computed on the real assembly.

The file is cached under `data/pdb/` with its SHA-256 recorded, so a reader can
verify that the numbers came from the structure we say they came from.

WHAT THIS IS NOT
----------------
This is a C-alpha model of one deposited structure. It is not a molecular
dynamics ensemble, not an atomic-resolution surface, and not a statement about
AAV9's conformational heterogeneity in solution. `what_this_structure_cannot_tell_you()`
(envelope limit)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The structure we actually use
# ---------------------------------------------------------------------------

AAV9_STRUCTURE_ID = "3UX1"
AAV9_STRUCTURE_TITLE = (
    "Structural characterization of adeno-associated virus serotype 9"
)
AAV9_STRUCTURE_REFERENCE = (
    "DiMattia MA, Nam HJ, Van Vliet K, et al. Structural insight into the "
    "unique properties of adeno-associated virus serotype 9. "
    "J Virol. 2012;86(12):6947-6958. PDB 3UX1."
)
AAV9_STRUCTURE_METHOD = "X-ray / cryo-EM-assisted model of the T=1 capsid"
AAV9_STRUCTURE_CHAIN_AA_RANGE = (219, 736)

#: What the repository previously cited. Kept as a named constant because the
#: correction is part of the project's record: anyone grepping for `3J1S` should
#: find the explanation, not a silently deleted string.
MISLABELED_STRUCTURE_ID = "3J1S"
MISLABELED_STRUCTURE_ACTUAL = (
    "PDB 3J1S is adeno-associated virus 2 in complex with neutralising "
    "monoclonal antibody A20 (McCraw DM, O'Donnell JK, Taylor KA, Stagg SM, "
    "Chapman MS. J Virol. 2012;86(23):12722-12730) -- serotype 2, not 9."
)

#: The AAV9 VP1 numbering offset this repository's design windows are expressed
#: in: stored sequence index = VP1 residue - VP1_OFFSET.
VP1_OFFSET = 263

#: Positions in the repository's *design* frame are 1-based, so that a position
#: can be used as a residue label directly: the residue edited is
#: `sequence[position - 1]`, and its VP1 number is `position + VP1_POSITION_OFFSET`.
#:
#: Both offsets exist because the coordinate tables were originally written with
#: table keys `vp1 - 262` while every consumer indexed `sequence[position - 1]`,
#: which silently edited the residue one position *before* the one it labelled.
#: The two conventions are named here so that mistake cannot recur.
VP1_POSITION_OFFSET = 262

#: Design windows, in **VP1 numbering** (== the deposited residue numbers of
#: 3UX1, whose chain A covers VP1 219-736).
VR_REGION_WINDOWS: Dict[str, Tuple[int, int]] = {
    "VR_IV": (449, 469),
    "VR_VIII": (571, 601),
}

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DEFAULT_STRUCTURE_DIR = os.path.join("data", "pdb")

#: SHA-256 of the cached `data/pdb/3UX1.pdb`, as downloaded from RCSB. Verified
#: on every load of the default path: a mismatch means the file is not the one
#: every published number in this repository was computed against, and the load
#: refuses rather than quietly reporting numbers from a different object.
EXPECTED_SHA256 = "1a39ed430f8404b955c7d58d1b66d34060398d0a32912a6cde11e11ee310e65f"


# ---------------------------------------------------------------------------
# Fetch / cache
# ---------------------------------------------------------------------------


def structure_path(
    structure_id: str = AAV9_STRUCTURE_ID,
    directory: str = DEFAULT_STRUCTURE_DIR,
) -> str:
    return os.path.join(directory, f"{structure_id.upper()}.pdb")


def ensure_structure_file(
    structure_id: str = AAV9_STRUCTURE_ID,
    directory: str = DEFAULT_STRUCTURE_DIR,
    *,
    timeout: float = 60.0,
    allow_download: bool = True,
) -> Tuple[str, bool]:
    """Return `(path, downloaded)`, fetching the deposited file if absent.

    Raises `FileNotFoundError` if the file is absent and downloading is
    disallowed or fails. It never falls back to embedded coordinates: a
    structural calculation without the structure is not a calculation this
    module is willing to perform.
    """
    path = structure_path(structure_id, directory)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, False
    if not allow_download:
        raise FileNotFoundError(
            f"{path} is missing and downloads are disabled; the capsid layer "
            f"needs the deposited coordinates"
        )
    os.makedirs(directory, exist_ok=True)
    url = RCSB_DOWNLOAD_URL.format(pdb_id=structure_id.upper())
    logger.info("downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = resp.read()
    except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - network
        raise FileNotFoundError(
            f"could not download {url}: {exc}; place {structure_id.upper()}.pdb "
            f"in {directory} manually"
        ) from exc
    if not payload.startswith(b"HEADER"):
        raise ValueError(f"{url} did not return a PDB file")
    with open(path, "wb") as fh:
        fh.write(payload)
    return path, True


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Parsing the deposit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaResidue:
    """One C-alpha atom from the deposited model."""

    resseq: int          # deposited residue number == VP1 numbering for 3UX1
    aa: str              # three-letter residue name
    x: float
    y: float
    z: float

    @property
    def vp1_position(self) -> int:
        return self.resseq

    @property
    def stored_position(self) -> int:
        """0-based index in this repository's stored VP1 sequence (offset 263)."""
        return self.resseq - VP1_OFFSET

    @property
    def design_position(self) -> int:
        """1-based design position: `sequence[design_position - 1]` is this residue."""
        return self.resseq - VP1_POSITION_OFFSET

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class SymmetryOperator:
    """One `REMARK 350 BIOMT` operator, with its rotation axis classified."""

    index: int
    matrix: np.ndarray          # 4x4
    angle_deg: float            # rotation angle implied by the matrix
    order: int                  # 1, 2, 3, or 5 (1 == identity)
    axis: Optional[np.ndarray]  # unit rotation axis (None for the identity)


_BIOMT_RE = re.compile(
    r"^REMARK 350\s+BIOMT(\d)\s+(\d+)\s+"
    r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)


def parse_ca_residues(path: str) -> List[CaResidue]:
    residues: List[CaResidue] = []
    with open(path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            residues.append(
                CaResidue(
                    resseq=int(line[22:26]),
                    aa=line[17:20].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                )
            )
    if not residues:
        raise ValueError(f"no CA atoms found in {path}")
    return residues


def parse_symmetry_operators(path: str) -> np.ndarray:
    """Return the deposited biological-assembly operators as an (N, 4, 4) array."""
    rows: Dict[int, np.ndarray] = {}
    with open(path, "r") as fh:
        for line in fh:
            m = _BIOMT_RE.match(line)
            if not m:
                continue
            row = int(m.group(1)) - 1
            op = int(m.group(2))
            rows.setdefault(op, np.zeros((4, 4), dtype=float))[row] = [
                float(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
                float(m.group(6)),
            ]
    if not rows:
        raise ValueError(f"no REMARK 350 BIOMT operators found in {path}")
    ops = np.empty((len(rows), 4, 4), dtype=float)
    for i, op in enumerate(sorted(rows)):
        block = rows[op]
        if not np.any(block):
            raise ValueError(f"BIOMT operator {op} in {path} has no matrix rows")
        ops[i, :3, :] = block[:3, :]
        ops[i, 3, :] = [0.0, 0.0, 0.0, 1.0]   # PDB writes only the top three rows
    return ops


def classify_symmetry_operators(ops: np.ndarray) -> List[SymmetryOperator]:
    """Label each operator by its rotation order, and return its axis.

    The classification is read off the deposited matrices: the rotation angle
    follows from the trace, and the rotation axis is the eigenvector with
    eigenvalue 1. A T=1 icosahedral assembly must come out as exactly
    1 identity, 15 two-fold, 20 three-fold, and 24 five-fold operators
    (12 x 72 deg + 12 x 144 deg) -- `CapsidAssembly` enforces that, so a
    corrupt or truncated operator list cannot pass silently.
    """
    out: List[SymmetryOperator] = []
    for i, m in enumerate(ops):
        R = m[:3, :3]
        cos_theta = (float(np.trace(R)) - 1.0) / 2.0
        angle = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
        if angle < 1e-3:
            out.append(SymmetryOperator(i, m, 0.0, 1, None))
            continue
        # Rotation order from the angle: 180 -> 2, 120 -> 3, 72/144 -> 5.
        if abs(angle - 180.0) < 1.0:
            order = 2
        elif abs(angle - 120.0) < 1.0:
            order = 3
        elif abs(angle - 72.0) < 1.0 or abs(angle - 144.0) < 1.0:
            order = 5
        else:
            raise ValueError(
                f"operator {i} has rotation angle {angle:.2f} deg, which is not "
                f"an icosahedral symmetry element"
            )
        w, v = np.linalg.eig(R)
        axis = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
        axis = axis / np.linalg.norm(axis)
        # Canonical sign so axes are reproducible across runs and platforms.
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        out.append(SymmetryOperator(i, m, angle, order, axis))
    return out


def axis_lines(operators: Sequence[SymmetryOperator], order: int) -> List[np.ndarray]:
    """Distinct rotation *lines* of a given order (antipodal axes merged).

    An icosahedral assembly has 6 five-fold lines, 10 three-fold lines and
    15 two-fold lines, even though it has 24 / 20 / 15 operators of those
    orders -- each line carries both signs of the rotation.
    """
    lines: List[np.ndarray] = []
    for op in operators:
        if op.order != order or op.axis is None:
            continue
        if any(abs(abs(float(op.axis @ ln)) - 1.0) < 1e-4 for ln in lines):
            continue
        lines.append(op.axis)
    return lines


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


class CapsidAssembly:
    """A real T=1 AAV9 capsid, built from the deposited coordinates and symmetry."""

    #: Icosahedral rotation group I (order 60) decomposes as these element counts.
    EXPECTED_OPERATOR_COUNTS = {1: 1, 2: 15, 3: 20, 5: 24}
    EXPECTED_LINE_COUNTS = {2: 15, 3: 10, 5: 6}

    def __init__(
        self,
        residues: Sequence[CaResidue],
        operators: np.ndarray,
        *,
        structure_id: str = AAV9_STRUCTURE_ID,
        sha256: Optional[str] = None,
    ) -> None:
        self.structure_id = structure_id
        self.sha256 = sha256
        self.residues: List[CaResidue] = list(residues)
        self.operators = np.asarray(operators, dtype=float)
        self.classified = classify_symmetry_operators(self.operators)

        counts: Dict[int, int] = {}
        for op in self.classified:
            counts[op.order] = counts.get(op.order, 0) + 1
        if counts != self.EXPECTED_OPERATOR_COUNTS:
            raise ValueError(
                f"symmetry operators do not form an icosahedral assembly: "
                f"got {counts}, expected {self.EXPECTED_OPERATOR_COUNTS}"
            )
        for order, expected in self.EXPECTED_LINE_COUNTS.items():
            got = len(axis_lines(self.classified, order))
            if got != expected:
                raise ValueError(
                    f"expected {expected} {order}-fold axis lines, got {got}"
                )

        self.ca = np.array([r.coord for r in self.residues], dtype=float)
        self.resseq = np.array([r.resseq for r in self.residues], dtype=int)

        # The full 60-subunit assembly, generated by the deposit's own operators.
        self.assembly = np.concatenate(
            [self.ca @ op.matrix[:3, :3].T + op.matrix[:3, 3] for op in self.classified]
        )
        self.center = self.assembly.mean(axis=0)
        radii = np.linalg.norm(self.assembly - self.center, axis=1)
        self.outer_radius = float(radii.max())
        self.mean_radius = float(radii.mean())
        self.inner_radius = float(radii.min())

        self._contacts: Optional[np.ndarray] = None

    # -- geometry ---------------------------------------------------------

    @property
    def subunit_count(self) -> int:
        return len(self.operators)

    @property
    def atom_count(self) -> int:
        return int(self.assembly.shape[0])

    def residue_radius(self) -> np.ndarray:
        """Radius of each asymmetric-unit residue from the capsid centre, in A."""
        return np.linalg.norm(self.ca - self.center, axis=1)

    def contact_numbers(self, cutoff: float = 10.0) -> np.ndarray:
        """Per-residue C-alpha contact count over the *whole* assembly.

        A real, derived burial measure: for each residue of the asymmetric unit,
        how many C-alpha atoms (across all 60 copies) lie within `cutoff` A.
        Higher == more buried. This is a contact number, not a solvent-
        accessible surface area -- computing SASA needs atomic radii and a
        probe, which this C-alpha model does not carry.
        """
        if self._contacts is not None:
            return self._contacts
        centres = self.ca.copy()
        counts = np.zeros(len(centres), dtype=float)
        for op in self.classified:
            transformed = self.ca @ op.matrix[:3, :3].T + op.matrix[:3, 3]
            d2 = (
                (centres[:, None, :] - transformed[None, :, :]) ** 2
            ).sum(axis=2)
            counts += (d2 < cutoff * cutoff).sum(axis=1)
        # each residue counts itself once, in the identity copy
        self._contacts = counts - 1.0
        return self._contacts

    def exposure_proxy(self) -> np.ndarray:
        """A documented proxy for surface exposure, in [0, 1].

        `radius / outer_radius` -- how far out along the capsid the residue's
        C-alpha sits. It is a radial exposure measure, NOT solvent
        accessibility, and it is named accordingly wherever it is used.
        """
        return self.residue_radius() / self.outer_radius

    # -- symmetry ---------------------------------------------------------

    def axes(self, order: int) -> List[np.ndarray]:
        return axis_lines(self.classified, order)

    @property
    def three_fold_axes(self) -> List[np.ndarray]:
        return self.axes(3)

    @property
    def five_fold_axes(self) -> List[np.ndarray]:
        return self.axes(5)

    @property
    def two_fold_axes(self) -> List[np.ndarray]:
        return self.axes(2)

    def protrusion_axes(self) -> List[np.ndarray]:
        """The 20 three-fold *operator* directions (the capsid's protrusions).

        Note the distinction that matters on a poster: AAV9 has 20 three-fold
        protrusions (one per three-fold operator position), which lie on
        10 three-fold axis *lines* because opposite pairs share a line.
        """
        return [op.axis for op in self.classified if op.order == 3 and op.axis is not None]

    # -- the design windows ----------------------------------------------

    def region_residues(self, region: str) -> List[CaResidue]:
        if region not in VR_REGION_WINDOWS:
            raise KeyError(f"unknown region {region!r}; known: {sorted(VR_REGION_WINDOWS)}")
        lo, hi = VR_REGION_WINDOWS[region]
        return [r for r in self.residues if lo <= r.resseq <= hi]

    def region_coordinate_table(self, region: str) -> Dict[int, Tuple[float, float, float, float]]:
        """`{design_position: (x, y, z, exposure_proxy)}` for a design window.

        Keys are **1-based** design positions (see `VP1_POSITION_OFFSET`), the
        frame every consumer indexes as `sequence[position - 1]`. Coordinates
        come from the deposit.
        """
        exposure = dict(zip(self.resseq.tolist(), self.exposure_proxy().tolist()))
        return {
            r.design_position: (r.x, r.y, r.z, float(exposure[r.resseq]))
            for r in self.region_residues(region)
        }

    def build_vr_coordinate_map(self) -> Dict[str, Dict]:
        """The real replacement for the fabricated `AAV9_PDB_3J1S_VR_COORDINATES`."""
        out: Dict[str, Dict] = {}
        for region in sorted(VR_REGION_WINDOWS):
            lo, hi = VR_REGION_WINDOWS[region]
            table = self.region_coordinate_table(region)
            radii = [float(np.linalg.norm(r.coord - self.center)) for r in self.region_residues(region)]
            out[region] = {
                "residues": sorted(table.keys()),
                "vp1_range": [lo, hi],
                "radius_from_centre_a": [min(radii), max(radii)],
                "coordinates": table,
                "position_frame": {
                    "points_are": "1-based design positions",
                    "index_offset": VP1_OFFSET,
                    "position_to_vp1_offset": VP1_POSITION_OFFSET,
                },
                "source": f"PDB {self.structure_id} chain A, VP1 {lo}-{hi}",
            }
        return out


# ---------------------------------------------------------------------------
# Cached loader
# ---------------------------------------------------------------------------

_ASSEMBLY_CACHE: Dict[str, CapsidAssembly] = {}


def load_aav9_capsid(
    path: Optional[str] = None,
    *,
    directory: str = DEFAULT_STRUCTURE_DIR,
    allow_download: bool = True,
) -> CapsidAssembly:
    """Load (and cache) the real AAV9 capsid assembly."""
    if path is None:
        path, _ = ensure_structure_file(
            AAV9_STRUCTURE_ID, directory, allow_download=allow_download
        )
    key = os.path.abspath(path)
    if key in _ASSEMBLY_CACHE:
        return _ASSEMBLY_CACHE[key]
    digest = file_sha256(path)
    if os.path.abspath(structure_path(AAV9_STRUCTURE_ID, directory)) == key:
        if digest != EXPECTED_SHA256:
            raise ValueError(
                f"{path} has SHA-256 {digest}, but this repository's published "
                f"numbers were computed on {EXPECTED_SHA256}. Re-download the "
                f"file, or pass an explicit path to analyse a different one."
            )
    residues = parse_ca_residues(path)
    operators = parse_symmetry_operators(path)
    assembly = CapsidAssembly(
        residues, operators, structure_id=AAV9_STRUCTURE_ID, sha256=digest
    )
    _ASSEMBLY_CACHE[key] = assembly
    return assembly


def vr_coordinate_map(
    path: Optional[str] = None, *, directory: str = DEFAULT_STRUCTURE_DIR
) -> Dict[str, Dict]:
    """Convenience accessor for the real VR coordinate tables."""
    return load_aav9_capsid(path, directory=directory).build_vr_coordinate_map()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def structure_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "ca_coordinates": EvidenceStatus.ESTABLISHED,
        "symmetry_operators": EvidenceStatus.ESTABLISHED,
        "assembly_geometry": EvidenceStatus.ESTABLISHED,
        "symmetry_axes": EvidenceStatus.ESTABLISHED,
        "exposure_proxy": EvidenceStatus.UNKNOWN,
        "region_boundaries": EvidenceStatus.UNKNOWN,
    }


def what_this_structure_cannot_tell_you() -> List[str]:
    return [
        "This is one deposited C-alpha model, not an ensemble: it cannot report "
        "conformational heterogeneity, loop disorder, or the distribution of "
        "VR-IV / VR-VIII geometries that actually exists in a particle population.",
        "`exposure_proxy` is radius / outer_radius, a radial measure. It is not "
        "solvent-accessible surface area and must not be reported as such.",
        "The VR-IV and VR-VIII window boundaries are the design windows used by "
        "this repository's capsid generator; the deposited model contains the "
        "residues, but the choice of where the loop 'starts' is a convention, "
        "and the capsid is only resolved to C-alpha here.",
        "Side-chain conformations, charge microenvironments, and glycan or "
        "solvent structure are absent from a C-alpha trace, so nothing about "
        "electrostatics at the atomic level can be read off this module.",
        "The engineered variants this repository designs are not in the deposit. "
        "Their coordinates are predictions from the sequence, not measurements.",
    ]
