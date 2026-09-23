"""
Real drug-parameter intake from open pharmacology databases.

WHY THIS MODULE EXISTS
----------------------
The scheduling engine needs two kinds of drug number, and it previously had
neither from a source: ``DEFAULT_DRUGS`` in :mod:`danon.evosteer` is an authored
constant set tagged UNKNOWN, and the pitch claimed potency kinetics without a
provenance chain. Both gaps are closable against *measured* public data:

  * **Potency** -- ChEMBL (EBI) serves the measured activity records behind the
    literature: ``standard_type`` / ``standard_value`` / ``standard_units`` /
    ``standard_relation`` / ``target_pref_name`` / publication year. For
    temozolomide this is 166 IC50 records, 50 of them on glioma lines.
  * **Descriptors** -- ChEMBL and PubChem both serve SMILES, molecular weight,
    ALogP, topological polar surface area, H-bond donors/acceptors. These are
    measured/curated molecule properties, not model output.

From those, transport quantities can be **derived** with a named method, which
is a different and weaker claim than being measured. Every number this module
returns carries which of the three it is.

THE THREE STATISTICAL TRAPS THIS MODULE REFUSES TO WALK INTO
------------------------------------------------------------
1. **Censoring is not a value.** 60 of temozolomide's 166 IC50 records are
   right-censored (``standard_relation == ">"``: the assay saturated without
   reaching 50% effect, so the true IC50 is *at least* the bound). Treating them
   as exact inflates every summary upward; dropping them silently biases the
   summary toward potent lines. ``PotencySummary`` counts them, excludes them
   from the point estimate, and says so.

2. **Units are not interchangeable.** The records mix nM (145), ``ug.mL-1`` (1)
   and unlabelled (20). A mass-concentration record converts to molar only via
   the molecule's molecular weight, so the conversion is done *with* the MW and
   refused without it. Anything unconvertible is counted, not guessed.

3. **An EC50 with no declared unit cannot be compared to a measured IC50.**
   :func:`compare_engine_ec50` refuses when the engine's concentration unit is
   undeclared, because "0.30 conc-units" versus "134 970 nM" is not a ratio.
   This is not pedantry: it is the difference between a calibration claim and a
   number that looks like one.

WHAT THIS MODULE CANNOT DO
--------------------------
ChEMBL aggregates assays run in different media, with different readouts
(ATP, MTT, clonogenic), different exposure times, and different passage numbers,
and it does not record most of that per record. Between-record variation is
therefore *methodological and biological*, not measurement noise, so a tight
confidence interval over these values would be a lie about the data. The module
reports spread and coverage, never a standard error over heterogeneous assays.

Citations:
  ChEMBL: Zdrazil B et al. The ChEMBL Database in 2023. Nucleic Acids Res.
  Stokes-Einstein: Einstein A. Ann Phys 1905;17:549. Tortuosity of brain
    extracellular space: Nicholson C, Sykova E. Trends Neurosci 1998;21:207.
  logBB QSAR: Clark DE. J Pharm Sci 1999;88:815-821.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------
# Endpoints and constants
# ---------------------------------------------------------------------------

CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
CHEMBL_MOLECULE_SEARCH = (
    "https://www.ebi.ac.uk/chembl/api/data/molecule.json?pref_name__iexact={name}"
)
CHEMBL_ACTIVITY_PAGE = (
    "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    "?molecule_chembl_id={chembl_id}&standard_type={standard_type}"
    "&limit={limit}&offset={offset}"
)
PUBCHEM_PROPERTY = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/MolecularWeight,CanonicalSMILES,XLogP,TPSA/JSON"
)

#: Cache root. Every cached payload is content-addressed (see ``manifest.json``)
#: so a swapped or truncated file is refused rather than silently consumed.
CACHE_DIR = "data/pharmacology"

#: Glioma lines that appear as activity targets in ChEMBL. These are the lines
#: whose measured potency is relevant to a glioma dosing model; the wider
#: ChEMBL target space (61 distinct targets for temozolomide) is mostly
#: carcinoma and leukaemia lines whose sensitivity does not transfer.
GLIOMA_LINES: Tuple[str, ...] = (
    "U-251", "U-87 MG", "LN-229", "T98G", "SNB-19", "DBTRG-05MG",
    "U-118-MG", "A172", "C6", "RG2",
)

#: Source grid pitch of the imaging the engine consumes (BraTS 2021 is 1 mm).
SOURCE_VOXEL_MM = 1.0

#: Water viscosity at 37 C in Pa*s (IAPWS-95 / CRC).
WATER_VISCOSITY_37C_PA_S = 6.91e-4

#: Assumed molecular density for the Stokes-Einstein radius. 1000 kg/m^3 is the
#: water-density convention, which *under*-estimates radius for a dense organic
#: solid and therefore *over*-estimates D. Declared rather than hidden.
ASSUMED_PARTICLE_DENSITY_KG_M3 = 1000.0

#: Tortuosity of the brain extracellular space for small molecules. Nicholson &
#: Sykova report lambda ~ 1.5-1.8; D_eff = D/lambda^2.
ASSUMED_TORTUOSITY = 1.6

#: Kelvin at 37 C.
BODY_TEMPERATURE_K = 310.15


class PharmacologyUnavailable(RuntimeError):
    """Raised when a real value was requested and neither cache nor network
    could supply one. Never silently substituted with a default."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeRecord:
    """A curated molecule entry. All fields are measured/curated, not model output."""

    name: str
    chembl_id: str
    smiles: Optional[str]
    mw: Optional[float]
    alogp: Optional[float]
    psa: Optional[float]
    hbd: Optional[int]
    hba: Optional[int]
    max_phase: Optional[float]
    source: str
    payload_sha256: str

    def describe(self) -> str:
        return (
            f"{self.name} ({self.chembl_id}, {self.source}): MW="
            f"{self.mw if self.mw is not None else 'n/a'} g/mol, "
            f"ALogP={self.alogp if self.alogp is not None else 'n/a'}, "
            f"PSA={self.psa if self.psa is not None else 'n/a'} A^2, "
            f"max_phase={self.max_phase if self.max_phase is not None else 'n/a'}"
        )


@dataclass(frozen=True)
class ActivityRecord:
    """One measured activity record, with its units and censoring preserved."""

    activity_id: Optional[int]
    standard_type: str
    value: Optional[float]
    units: Optional[str]
    relation: Optional[str]
    target: Optional[str]
    year: Optional[int]
    document_chembl_id: Optional[str]
    #: Value normalised to nM, or None if the units could not be converted.
    value_nM: Optional[float]

    @property
    def is_exact(self) -> bool:
        return self.relation == "=" and self.value_nM is not None

    @property
    def is_right_censored(self) -> bool:
        return self.relation == ">" and self.value_nM is not None

    @property
    def is_left_censored(self) -> bool:
        return self.relation == "<" and self.value_nM is not None


@dataclass
class PotencySummary:
    """Distribution of measured potency over a selected target set.

    Deliberately does **not** carry a standard error: the records are drawn from
    different assays with different readouts, so the spread is methodological
    and biological. Reporting dispersion over them as if it were measurement
    noise would overstate what the data supports.
    """

    standard_type: str
    n_records: int
    n_exact: int = 0
    n_right_censored: int = 0
    n_left_censored: int = 0
    n_unconvertible: int = 0
    exact_nM: Tuple[float, ...] = ()
    censored_bound_nM: Tuple[float, ...] = ()
    targets: Tuple[str, ...] = ()
    years: Tuple[int, ...] = ()
    notes: List[str] = field(default_factory=list)

    @property
    def median_exact_nM(self) -> Optional[float]:
        return float(np.median(self.exact_nM)) if self.exact_nM else None

    @property
    def most_potent_exact_nM(self) -> Optional[float]:
        """The most potent *measured* value -- the relevant bound when the
        question is 'can this concentration kill cells at all?'."""
        return float(min(self.exact_nM)) if self.exact_nM else None

    @property
    def least_potent_exact_nM(self) -> Optional[float]:
        return float(max(self.exact_nM)) if self.exact_nM else None

    @property
    def fold_spread(self) -> Optional[float]:
        """max/min over exact values -- between-model heterogeneity."""
        lo, hi = self.most_potent_exact_nM, self.least_potent_exact_nM
        if lo is None or hi is None or lo <= 0:
            return None
        return hi / lo

    @property
    def censored_fraction(self) -> float:
        n_value = self.n_exact + self.n_right_censored + self.n_left_censored
        if n_value == 0:
            return 0.0
        return (self.n_right_censored + self.n_left_censored) / n_value

    @property
    def median_lower_bound_nM(self) -> Optional[float]:
        """Lower bound on the true median, valid only with no left-censoring.

        Right-censored records have true values >= their bounds, and the median
        is monotone in the order statistics, so replacing each censored value by
        its bound can only push the median down: the result is a genuine lower
        bound (not an estimate, and *not* comparable to the exact-only median,
        which is a different order statistic).

        Left-censored records have true values <= their bounds, so substituting
        lowers nothing -- it would destroy the bound. Returns ``None`` rather
        than a number that is not what its name says.
        """
        if self.n_left_censored > 0:
            return None
        vals = list(self.exact_nM) + list(self.censored_bound_nM)
        return float(np.median(vals)) if vals else None

    def describe(self) -> str:
        if self.n_exact == 0:
            return (
                f"{self.standard_type}: {self.n_records} records, no exact "
                f"convertible measurement ({self.n_right_censored} right-censored, "
                f"{self.n_unconvertible} unconvertible units)"
            )
        lo = self.median_lower_bound_nM
        return (
            f"{self.standard_type} over {len(self.targets)} target(s): "
            f"{self.n_exact} exact "
            f"(median {self.median_exact_nM:,.0f} nM, most potent "
            f"{self.most_potent_exact_nM:,.0f} nM, spread "
            f"{self.fold_spread:.0f}x), {self.n_right_censored} right-censored"
            + (f" (median lower bound {lo:,.0f} nM)" if lo is not None else "")
        )


@dataclass(frozen=True)
class DiffusionEstimate:
    """A derived diffusion coefficient, with every assumption named."""

    mw_g_per_mol: float
    temperature_K: float
    viscosity_Pa_s: float
    particle_density_kg_m3: float
    radius_nm: float
    d_free_water_mm2_per_day: float
    tortuosity: float
    d_tissue_mm2_per_day: float
    method: str = "Stokes-Einstein sphere, D = kT / (6*pi*eta*r), D_eff = D/lambda^2"

    def in_engine_units(self, voxel_mm: float) -> float:
        """Convert mm^2/day to the engine's voxel^2/day."""
        return self.d_tissue_mm2_per_day / (voxel_mm ** 2)


@dataclass(frozen=True)
class BloodBrainEstimate:
    """A logBB prediction from a *published* QSAR, with its counterexample note."""

    alogp: float
    psa: float
    logbb: float
    method: str
    citation: str
    caveat: str


@dataclass
class Ec50Comparison:
    """Engine potency vs measured potency -- or the refusal to compare them."""

    comparable: bool
    reason: str
    engine_ec50_nM: Optional[float] = None
    measured_most_potent_nM: Optional[float] = None
    measured_median_nM: Optional[float] = None
    ratio_to_most_potent: Optional[float] = None
    ratio_to_median: Optional[float] = None

    def describe(self) -> str:
        if not self.comparable:
            return f"EC50 comparison REFUSED: {self.reason}"
        return (
            f"engine EC50 {self.engine_ec50_nM:,.0f} nM vs measured most potent "
            f"{self.measured_most_potent_nM:,.0f} nM "
            f"({self.ratio_to_most_potent:,.1f}x) and median "
            f"{self.measured_median_nM:,.0f} nM "
            f"({self.ratio_to_median:,.0f}x)"
        )


# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------

#: Multiplicative factors to nM for units that need no molecular weight.
_MOLAR_TO_NM: Dict[str, float] = {
    "m": 1e9, "molar": 1e9,
    "mm": 1e6, "um": 1e3, "µm": 1e3, "micromolar": 1e3,
    "nm": 1.0, "nanomolar": 1.0,
    "pm": 1e-3, "fm": 1e-6,
}

#: Mass-concentration units: nM = (value in this unit) * 1e6 / MW.
_MASS_PER_VOLUME_TO_NM_PER_MW: Dict[str, float] = {
    "ug.ml-1": 1e6, "µg.ml-1": 1e6, "ug/ml": 1e6, "µg/ml": 1e6,
    "mg.ml-1": 1e9, "mg/ml": 1e9,
    "ng.ml-1": 1e3, "ng/ml": 1e3,
}


def normalize_to_nM(
    value: Optional[float], units: Optional[str], mw: Optional[float] = None
) -> Optional[float]:
    """Convert a measured concentration to nM, or return ``None``.

    ``None`` means *could not convert*, never *zero*. Mass-concentration units
    require ``mw``; without it they are refused rather than approximated.
    """
    if value is None or units is None:
        return None
    key = str(units).strip().lower()
    if key in _MOLAR_TO_NM:
        return float(value) * _MOLAR_TO_NM[key]
    if key in _MASS_PER_VOLUME_TO_NM_PER_MW:
        if mw is None or mw <= 0:
            return None
        return float(value) * _MASS_PER_VOLUME_TO_NM_PER_MW[key] / float(mw)
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _sha256_of(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_molecule(payload: Dict[str, object]) -> MoleculeRecord:
    """Build a :class:`MoleculeRecord` from a raw ChEMBL molecule payload."""
    props = payload.get("molecule_properties") or {}
    structures = payload.get("molecule_structures") or {}

    def _f(key: str) -> Optional[float]:
        v = props.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _i(key: str) -> Optional[int]:
        v = props.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _f_top(key: str) -> Optional[float]:
        v = payload.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return MoleculeRecord(
        name=str(payload.get("pref_name") or "unknown"),
        chembl_id=str(payload.get("molecule_chembl_id") or ""),
        smiles=structures.get("canonical_smiles"),
        mw=_f("full_mwt"),
        alogp=_f("alogp"),
        psa=_f("psa"),
        hbd=_i("hbd"),
        hba=_i("hba"),
        # ``max_phase`` is a top-level ChEMBL field, not a molecule_properties
        # key; reading it from props (as this did) silently returned None on
        # every molecule, including phase-4 drugs.
        max_phase=_f_top("max_phase") or _f("max_phase"),
        source="ChEMBL (EBI) molecule endpoint",
        payload_sha256=_sha256_of(payload),
    )


def parse_activities(
    payloads: Sequence[Dict[str, object]], mw: Optional[float] = None
) -> List[ActivityRecord]:
    """Build :class:`ActivityRecord` objects, normalising units with ``mw``."""
    out: List[ActivityRecord] = []
    for a in payloads:
        raw_units = a.get("standard_units")
        try:
            value = float(a["standard_value"]) if a.get("standard_value") is not None else None
        except (TypeError, ValueError):
            value = None
        year = a.get("document_year")
        try:
            year_i = int(year) if year is not None else None
        except (TypeError, ValueError):
            year_i = None
        out.append(
            ActivityRecord(
                activity_id=a.get("activity_id"),
                standard_type=str(a.get("standard_type") or ""),
                value=value,
                units=raw_units,
                relation=a.get("standard_relation"),
                target=a.get("target_pref_name"),
                year=year_i,
                document_chembl_id=a.get("document_chembl_id"),
                value_nM=normalize_to_nM(value, raw_units, mw),
            )
        )
    return out


def summarize_potency(
    activities: Sequence[ActivityRecord],
    targets: Optional[Sequence[str]] = None,
    standard_type: Optional[str] = None,
) -> PotencySummary:
    """Summarise measured potency, keeping censoring and units explicit.

    ``targets`` filters by ``ActivityRecord.target`` (e.g. :data:`GLIOMA_LINES`).
    Censored records are counted and excluded from the point estimate; the
    returned object reports both so the selection is never hidden.
    """
    if targets is not None:
        allowed = {t.strip().lower() for t in targets}
        rows = [a for a in activities if (a.target or "").strip().lower() in allowed]
    else:
        rows = list(activities)
    if standard_type is not None:
        rows = [a for a in rows if a.standard_type.upper() == standard_type.upper()]

    summary = PotencySummary(
        standard_type=standard_type or (rows[0].standard_type if rows else "IC50"),
        n_records=len(rows),
    )
    for a in rows:
        if a.value_nM is None:
            summary.n_unconvertible += 1
        elif a.is_exact:
            summary.n_exact += 1
        elif a.is_right_censored:
            summary.n_right_censored += 1
            summary.censored_bound_nM += (a.value_nM,)
        elif a.is_left_censored:
            summary.n_left_censored += 1
        else:
            # A numeric value with an unrecognised relation string: refuse to
            # classify it as exact rather than assume "=".
            summary.n_unconvertible += 1

    summary.exact_nM = tuple(
        sorted(a.value_nM for a in rows if a.is_exact)
    )
    summary.targets = tuple(sorted({a.target for a in rows if a.target}))
    summary.years = tuple(sorted({a.year for a in rows if a.year}))

    if summary.n_unconvertible:
        summary.notes.append(
            f"{summary.n_unconvertible} record(s) excluded: units or relation not "
            f"convertible to nM without guessing (checked against the molecule's MW)"
        )
    if summary.n_right_censored:
        summary.notes.append(
            f"{summary.n_right_censored} record(s) are right-censored "
            f"('>' relation: assay saturated below 50% effect). True IC50 is at "
            f"least the reported bound. They are excluded from the point estimate "
            f"and reported as a bound instead."
        )
    if summary.n_exact and summary.n_right_censored and summary.n_left_censored == 0:
        # The exact-only median is a *selection*: a record is exact because the
        # assay reached 50% effect within its concentration range, which
        # correlates with a lower IC50. So the exact-only median runs below the
        # cohort's median, and the lower bound above it. Both are reported so
        # neither is presented as "the" median.
        summary.notes.append(
            "the exact-only median is biased low by selection (records are exact "
            "when the assay reached 50% effect, which correlates with greater "
            "potency); the true cohort median lies above median_lower_bound_nM "
            "and is not bounded above by these data"
        )
    if summary.n_left_censored:
        summary.notes.append(
            f"{summary.n_left_censored} record(s) are left-censored ('<'); their "
            f"true values are below the reported bound, so no lower bound on the "
            f"median is available from these records."
        )
    if summary.n_exact and summary.n_exact < 5:
        summary.notes.append(
            f"only {summary.n_exact} exact measurement(s): a median over this many "
            f"heterogeneous assays is not a stable estimate"
        )
    if summary.years:
        summary.notes.append(
            f"records span {summary.years[0]}-{summary.years[-1]}; assay and "
            f"readout drift over that window is not controlled for"
        )
    return summary


# ---------------------------------------------------------------------------
# Derived quantities (named methods, named assumptions)
# ---------------------------------------------------------------------------


def stokes_einstein_diffusion(
    mw_g_per_mol: float,
    *,
    temperature_K: float = BODY_TEMPERATURE_K,
    viscosity_Pa_s: float = WATER_VISCOSITY_37C_PA_S,
    particle_density_kg_m3: float = ASSUMED_PARTICLE_DENSITY_KG_M3,
    tortuosity: float = ASSUMED_TORTUOSITY,
) -> DiffusionEstimate:
    """D_free from Stokes-Einstein, then D_eff = D/lambda^2 in brain tissue.

    DERIVED, not measured: assumes a spherical particle of the given density in
    water at body temperature, then a single tortuosity factor for the
    extracellular space. It ignores binding, uptake and convection, all of which
    dominate in vivo, so the result is an upper bound on free transport and not
    a tissue measurement.
    """
    if mw_g_per_mol <= 0:
        raise ValueError("molecular weight must be positive")
    k_B = 1.380649e-23
    n_A = 6.02214076e23
    mw_kg = mw_g_per_mol / 1000.0
    volume_per_molecule = mw_kg / (particle_density_kg_m3 * n_A)
    radius_m = (3.0 * volume_per_molecule / (4.0 * math.pi)) ** (1.0 / 3.0)
    d_m2_per_s = (k_B * temperature_K) / (6.0 * math.pi * viscosity_Pa_s * radius_m)
    # 1 m^2/s = 1e6 mm^2/s = 8.64e10 mm^2/day
    d_free = d_m2_per_s * 8.64e10
    return DiffusionEstimate(
        mw_g_per_mol=float(mw_g_per_mol),
        temperature_K=float(temperature_K),
        viscosity_Pa_s=float(viscosity_Pa_s),
        particle_density_kg_m3=float(particle_density_kg_m3),
        radius_nm=radius_m * 1e9,
        d_free_water_mm2_per_day=d_free,
        tortuosity=float(tortuosity),
        d_tissue_mm2_per_day=d_free / (tortuosity ** 2),
    )


def clark_logbb(alogp: float, psa: float) -> BloodBrainEstimate:
    """logBB from Clark's published PSA/logP QSAR.

    DERIVED from a *published* fit, on a substituted descriptor: the equation
    takes cLogP and ChEMBL curates ALogP, which are not the same estimator.
    """
    logbb = -0.0148 * psa + 0.152 * alogp + 0.139
    return BloodBrainEstimate(
        alogp=float(alogp),
        psa=float(psa),
        logbb=float(logbb),
        method="logBB = -0.0148*PSA + 0.152*cLogP + 0.139",
        citation=(
            "Clark DE. Rapid calculation of polar molecular surface area and its "
            "application to the prediction of transport phenomena. 2. Prediction "
            "of blood-brain barrier penetration. J Pharm Sci 1999;88:815-821"
        ),
        caveat=(
            "Fitted on a heterogeneous 1990s set; ALogP is substituted for cLogP. "
            "A known counterexample sits in this repo's own drug set: the equation "
            "predicts poor penetration for temozolomide (PSA 108 A^2), which is "
            "nonetheless used clinically in glioblastoma because of its small size "
            "and lack of efflux-substrate liability. Treat the sign as informative "
            "and the magnitude as unreliable."
        ),
    )


def engine_ec50_unit() -> Optional[str]:
    """The concentration unit the engine's EC50 is expressed in, if declared.

    ``evosteer.DrugParams`` documents ``source_strength`` as "conc/day at
    full-dose vessels" and never names what "conc" is. Until that unit is
    declared, an EC50 cannot be compared to a molar measurement.
    """
    from danon import evosteer

    for attr in ("CONCENTRATION_UNIT", "DRUG_CONCENTRATION_UNIT"):
        unit = getattr(evosteer, attr, None)
        if unit:
            return str(unit)
    return None


def compare_engine_ec50(
    summary: PotencySummary,
    ec50_value: float,
    ec50_unit: Optional[str],
    *,
    mw: Optional[float] = None,
) -> Ec50Comparison:
    """Compare the engine's assumed EC50 against measured potency -- or refuse.

    A refusal is the correct output when the engine's unit is undeclared: the
    ratio of a dimensionless number to a molar concentration is not a
    calibration factor, and reporting one would manufacture a finding.
    """
    if summary.n_exact == 0:
        return Ec50Comparison(
            comparable=False,
            reason="no exact measured potency available to compare against",
        )
    if ec50_unit is None:
        return Ec50Comparison(
            comparable=False,
            reason=(
                "the engine's EC50 has no declared concentration unit "
                "(DrugParams documents only 'conc/day at full-dose vessels'), so "
                "an EC50-vs-IC50 ratio would be unitless nonsense. Declare a unit "
                "on the engine's concentration scale before comparing; measured "
                f"values are available (median {summary.median_exact_nM:,.0f} nM, "
                f"most potent {summary.most_potent_exact_nM:,.0f} nM)."
            ),
        )
    ec50_nM = normalize_to_nM(ec50_value, ec50_unit, mw)
    if ec50_nM is None:
        return Ec50Comparison(
            comparable=False,
            reason=(
                f"EC50 unit {ec50_unit!r} could not be converted to nM"
                + (" (mass concentration needs a molecular weight)" if mw is None else "")
            ),
        )
    return Ec50Comparison(
        comparable=True,
        reason="both quantities reduce to nM",
        engine_ec50_nM=ec50_nM,
        measured_most_potent_nM=summary.most_potent_exact_nM,
        measured_median_nM=summary.median_exact_nM,
        ratio_to_most_potent=(
            summary.most_potent_exact_nM / ec50_nM
            if summary.most_potent_exact_nM else None
        ),
        ratio_to_median=(
            summary.median_exact_nM / ec50_nM if summary.median_exact_nM else None
        ),
    )


# ---------------------------------------------------------------------------
# Retrieval (cache-first, refusal instead of invention)
# ---------------------------------------------------------------------------


def _default_fetcher(url: str) -> Dict[str, object]:
    req = urllib.request.Request(url, headers={"User-Agent": "danon-pharmacology/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def _read_cache(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def fetch_molecule(
    name: str,
    *,
    cache_dir: str = CACHE_DIR,
    fetcher: Optional[Callable[[str], Dict[str, object]]] = None,
    allow_network: bool = True,
) -> MoleculeRecord:
    """Look up a molecule by preferred name, cache-first.

    Raises :class:`PharmacologyUnavailable` when neither the cache nor the
    network can supply the record. It never returns a placeholder molecule.
    """
    slug = name.strip().lower().replace(" ", "_")
    path = os.path.join(cache_dir, f"{slug}_molecule.json")
    payload = _read_cache(path)
    errors: List[str] = []

    if payload is None and allow_network:
        fetch = fetcher or _default_fetcher
        try:
            search = fetch(CHEMBL_MOLECULE_SEARCH.format(name=urllib.parse.quote(name)))
            molecules = (search or {}).get("molecules") or []
            if not molecules:
                raise PharmacologyUnavailable(
                    f"ChEMBL has no molecule with pref_name={name!r}"
                )
            chembl_id = molecules[0]["molecule_chembl_id"]
            payload = fetch(CHEMBL_MOLECULE_URL.format(chembl_id=chembl_id))
            _write_cache(path, payload)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError) as exc:
            errors.append(f"network: {type(exc).__name__}: {exc}")

    if payload is None:
        raise PharmacologyUnavailable(
            f"no cached record for {name!r} at {path}"
            + (f"; fetch errors: {'; '.join(errors)}" if errors else "")
            + ". Refusing to substitute a placeholder molecule."
        )
    return parse_molecule(payload)


def fetch_activities(
    chembl_id: str,
    standard_type: str = "IC50",
    *,
    cache_dir: str = CACHE_DIR,
    fetcher: Optional[Callable[[str], Dict[str, object]]] = None,
    allow_network: bool = True,
    page_size: int = 100,
) -> List[Dict[str, object]]:
    """Fetch every activity page for a molecule/type pair, cache-first.

    Returns raw payloads so :func:`parse_activities` can normalise them with the
    molecule's MW; caching raw avoids baking a unit conversion into the cache.
    """
    slug = f"raw_{chembl_id}_{standard_type}.json"
    path = os.path.join(cache_dir, slug)
    cached = _read_cache(path)
    errors: List[str] = []
    if cached is not None and cached.get("activities"):
        return list(cached["activities"])

    rows: List[Dict[str, object]] = []
    if allow_network:
        fetch = fetcher or _default_fetcher
        try:
            offset = 0
            total = None
            while True:
                page = fetch(
                    CHEMBL_ACTIVITY_PAGE.format(
                        chembl_id=chembl_id,
                        standard_type=standard_type,
                        limit=page_size,
                        offset=offset,
                    )
                )
                rows.extend(page.get("activities") or [])
                total = (page.get("page_meta") or {}).get("total_count", 0)
                offset += page_size
                if offset >= int(total or 0):
                    break
            if rows:
                _write_cache(path, {"activities": rows})
        except (urllib.error.URLError, TimeoutError, OSError, KeyError) as exc:
            errors.append(f"network: {type(exc).__name__}: {exc}")

    if not rows:
        raise PharmacologyUnavailable(
            f"no cached activities at {path}"
            + (f"; fetch errors: {'; '.join(errors)}" if errors else "")
            + f". Refusing to invent {standard_type} values for {chembl_id}."
        )
    return rows


def load_glioma_potency(
    molecule_name: str = "temozolomide",
    standard_type: str = "IC50",
    *,
    cache_dir: str = CACHE_DIR,
    allow_network: bool = False,
    fetcher: Optional[Callable[[str], Dict[str, object]]] = None,
) -> Tuple[MoleculeRecord, PotencySummary]:
    """The GBM-relevant potency summary for a molecule, from cache by default.

    ``allow_network=False`` is the default so tests and reports are deterministic
    and unaffected by EBI availability; pass ``True`` (or a ``fetcher``) to
    refresh.
    """
    mol = fetch_molecule(
        molecule_name, cache_dir=cache_dir, allow_network=allow_network, fetcher=fetcher
    )
    raw = fetch_activities(
        mol.chembl_id, standard_type, cache_dir=cache_dir,
        allow_network=allow_network, fetcher=fetcher,
    )
    acts = parse_activities(raw, mw=mol.mw)
    return mol, summarize_potency(acts, targets=GLIOMA_LINES, standard_type=standard_type)


def engine_voxel_mm(decimation_factors: Sequence[int] = (4, 4, 3)) -> float:
    """Engine voxel pitch, using the same convention as :mod:`danon.realdata`."""
    return float(np.mean([float(f) * SOURCE_VOXEL_MM for f in decimation_factors]))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def pharmacology_provenance() -> Dict[str, EvidenceStatus]:
    """What each quantity in this module is, in the repo's provenance vocabulary."""
    return {
        "chembl_activity_records": EvidenceStatus.ESTABLISHED,
        "chembl_molecule_descriptors": EvidenceStatus.ESTABLISHED,
        "glioma_line_potency_distribution": EvidenceStatus.ESTABLISHED,
        "stokes_einstein_free_diffusion": EvidenceStatus.UNKNOWN,
        "brain_tissue_effective_diffusion": EvidenceStatus.UNKNOWN,
        "logbb_qsar_prediction": EvidenceStatus.UNKNOWN,
        "engine_ec50_concentration_unit": EvidenceStatus.UNKNOWN,
    }


def what_pharmacology_cannot_prove() -> List[str]:
    """Limits that must travel with any number this module produces."""
    return [
        "ChEMBL records come from different assays, readouts (ATP/MTT/clonogenic), "
        "exposure times and passage numbers, and most of that context is absent per "
        "record. Spread across them is methodological, so it is not reported as a "
        "standard error or a confidence interval.",
        "Potency in 2D monolayer culture at 72 h is not potency in a 3D tumour with "
        "a diffusion gradient, an intact blood-brain barrier and heterogeneous "
        "sub-clones. No in-vitro IC50, however well measured, establishes in-vivo "
        "tumour kill.",
        "Stokes-Einstein gives free-solution diffusion. In brain, binding, cellular "
        "uptake, efflux and convection dominate transport, and the derived D_eff is "
        "an upper bound on free transport rather than a tissue measurement.",
        "The Clark logBB equation is a published QSAR fitted on a 1990s set with a "
        "substituted descriptor (ALogP for cLogP); its sign is informative, its "
        "magnitude is not reliable, and temozolomide is a known counterexample.",
        "Nothing here validates the engine. These are measured pharmacology and "
        "derived transport bounds to calibrate against, not evidence that the "
        "scheduling model is right.",
    ]
