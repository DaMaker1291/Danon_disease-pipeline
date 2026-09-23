"""
MGMT promoter methylation: the real betas, and how they enter the model.

WHY THIS MODULE EXISTS
----------------------
The proposed upgrade is stated as ``r0 = Baseline_Resistance * (1 - beta)``,
where ``beta`` is the patient's MGMT promoter methylation. Two things have to
be true before that line means anything: the beta has to be a real measurement
for real patients, and the mapping has to be stated in terms of a parameter the
engine actually has. This module supplies both, and measures the consequence of
the second rather than asserting it.

WHERE THE BETAS COME FROM
-------------------------
`danon.cohort_features.extract_mgmt_beta` reads a GDC `Methylation Beta Value`
file, which is the ground truth, but it is one 13 MB file per patient: the 142
TCGA-GBM patients on the 450k platform would cost ~1.8 GB to read. The same
beta values are published per probe through the cBioPortal generic-assay API,
so `load_mgmt_beta_panel` fetches the **canonical STP27 probe pair**
(`cg12434587`, `cg12981137` -- the same probes `cohort_features` uses) for every
PanCancer-Atlas GBM sample in a single small request, and caches the result.
The probe selection and the 0.358 cutoff are literature parameters, not
file-derived facts, and are marked as such.

The binary call is fetched too, from two published sources
(`MGMT_PROMOTER_STATUS` in the Cell 2016 pan-glioma cohort and `MGMT_STATUS` in
the Cell 2013 GBM cohort), because a binary call and a continuous beta are not
interchangeable and the disagreement between them is worth measuring:
`mgmt_call_concordance` recomputes the call from the retrieved betas with this
repository's own cutoff and compares it against what the publications report
for the same patients.

WHAT THE RESISTANCE MAPPING CAN AND CANNOT MOVE
-----------------------------------------------
`CloneParams` carries two separate components of drug-A resistance: the clone's
own kill ceiling (``DrugParams.kill_max_R[0]`` against ``kill_max_S[0]``) and a
multiplicative scale (``drug_sensitivity_scale_R``). The requested term is a
mapping from methylation to *a* resistance, so it is wired to the scale, with
the calibrated default defined as the unmethylated baseline. That choice bounds
the effect: `resistance_audit` computes the reachable range and it is small,
because most of the engine's resistance lives in the kill ceiling, not the
scale. A term that looks decisive in the formula is therefore modest in the
simulation, and the audit says so with numbers instead of adjectives.

References:
  Bady P et al. Acta Neuropathol 2012;124(4):547-560 (MGMT-STP27 probe pair and
  the average-beta threshold).
  Ceccarelli M et al. Cell 2016;164(3):550-563 (pan-glioma MGMT status).
  Brennan CW et al. Cell 2013;155(2):462-477 (TCGA GBM MGMT status).
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, fields, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from danon.cohort_features import MGMT_METHYLATION_CUTOFF, MGMT_PROBES
from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    CloneParams,
    TumorEvolutionSimulator,
    TumorGeometry,
    make_gbm_phantom,
    run_trial,
)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

CBIOPORTAL_API = "https://www.cbioportal.org/api"

#: PanCancer-Atlas GBM HM450 profile. Its stable IDs are probe IDs, and its
#: metadata records a gene symbol per probe, so the canonical STP27 pair is
#: addressable directly.
MGMT_BETA_PROFILE = "gbm_tcga_pan_can_atlas_2018_methylation_hm450"
MGMT_BETA_STUDY = "gbm_tcga_pan_can_atlas_2018"
MGMT_BETA_CACHE = "data/gdc_methylation/cbioportal_mgmt_probes.json"

#: Published binary calls: (study, clinical attribute).
MGMT_CALL_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("lgggbm_tcga_pub", "MGMT_PROMOTER_STATUS"),
    ("gbm_tcga_pub2013", "MGMT_STATUS"),
)
MGMT_CALL_CACHE = "data/gdc_clinical/cbioportal_mgmt_calls.json"

#: Call vocabularies that mean methylated / unmethylated. Anything else is
#: missing rather than coerced.
METHYLATED_TOKENS = frozenset({"methylated"})
UNMETHYLATED_TOKENS = frozenset({"unmethylated"})


def _default_fetcher(url: str, timeout: int = 120) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Continuous beta panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MgmtBetaPanel:
    """Per-patient probe betas read from the published per-probe endpoint."""

    profile: str
    probes: Dict[str, str]
    betas: Dict[str, Dict[str, float]]
    source: str
    n_records: int
    notes: List[str]

    @property
    def patient_ids(self) -> List[str]:
        return sorted(self.betas)

    @property
    def n_patients(self) -> int:
        return len(self.betas)

    def missing_probes(self, patient: str) -> List[str]:
        got = self.betas.get(patient, {})
        return [p for p in self.probes if p not in got]

    def complete_patients(self) -> List[str]:
        return [p for p in self.patient_ids if not self.missing_probes(p)]

    def mean_beta(self, patient: str) -> Optional[float]:
        """Mean over the probes that are present. Missing probes are not imputed."""
        got = [v for v in self.betas.get(patient, {}).values() if 0.0 <= v <= 1.0]
        return float(np.mean(got)) if got else None

    def call(self, patient: str, *, cutoff: float = MGMT_METHYLATION_CUTOFF) -> str:
        mean = self.mean_beta(patient)
        if mean is None:
            return "not callable"
        return "methylated" if mean >= cutoff else "unmethylated"

    def call_counts(self, *, cutoff: float = MGMT_METHYLATION_CUTOFF) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for patient in self.patient_ids:
            key = self.call(patient, cutoff=cutoff)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def betas_of(self, patients: Iterable[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for pid in patients:
            mean = self.mean_beta(pid)
            if mean is not None:
                out[pid] = mean
        return out

    def quantiles(self, qs: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> List[float]:
        values = [v for v in (self.mean_beta(p) for p in self.patient_ids) if v is not None]
        if not values:
            return []
        arr = np.asarray(values, dtype=float)
        return [float(np.quantile(arr, q)) for q in qs]

    def __str__(self) -> str:
        counts = self.call_counts()
        calls = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return (
            f"MGMT beta panel: {self.n_patients} patient(s) over "
            f"{len(self.probes)} probe(s) [{', '.join(sorted(self.probes))}], "
            f"{self.n_records} record(s); calls at cutoff "
            f"{MGMT_METHYLATION_CUTOFF:.3f}: {calls}"
        )


def load_mgmt_beta_panel(
    *,
    profile: str = MGMT_BETA_PROFILE,
    probes: Optional[Dict[str, str]] = None,
    fetcher: Optional[Callable[[str], object]] = None,
    cache_path: str = MGMT_BETA_CACHE,
    use_cache: bool = True,
) -> MgmtBetaPanel:
    """Fetch the canonical MGMT probe betas for every sample in the profile.

    Betas outside [0, 1] are dropped rather than clamped: a value outside the
    definition of a beta is a parsing or platform problem, and clamping it would
    invent a measurement. Probes absent for a patient stay absent.
    """
    probe_set = dict(probes or MGMT_PROBES)
    if use_cache:
        try:
            with open(cache_path) as fh:
                blob = json.load(fh)
            if blob.get("profile") == profile and blob.get("probes") == probe_set:
                return MgmtBetaPanel(
                    profile=blob["profile"],
                    probes=probe_set,
                    betas={k: dict(v) for k, v in blob["betas"].items()},
                    source=blob["source"],
                    n_records=int(blob["n_records"]),
                    notes=list(blob.get("notes") or []),
                )
        except (OSError, ValueError, TypeError, KeyError):
            pass

    fetch = fetcher or _default_fetcher
    betas: Dict[str, Dict[str, float]] = {}
    n_records = 0
    dropped_out_of_range = 0
    per_probe_counts: Dict[str, int] = {}
    for probe in probe_set:
        url = f"{CBIOPORTAL_API}/generic-assay-data/{profile}/generic-assay/{probe}"
        payload = fetch(url)
        records = payload if isinstance(payload, list) else []
        per_probe_counts[probe] = len(records)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            pid = rec.get("patientId")
            raw = rec.get("value")
            if not pid or raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            n_records += 1
            if not (0.0 <= value <= 1.0):
                dropped_out_of_range += 1
                continue
            betas.setdefault(str(pid), {})[probe] = value

    notes = [
        f"probe selection {sorted(probe_set)} is literature-derived (Bady 2012), "
        f"not derived from the file; a different probe set gives a different beta",
        f"records per probe: {per_probe_counts}",
    ]
    if dropped_out_of_range:
        notes.append(
            f"{dropped_out_of_range} value(s) fell outside [0, 1] and were dropped, "
            f"not clamped"
        )
    panel = MgmtBetaPanel(
        profile=profile,
        probes=probe_set,
        betas=betas,
        source=f"{CBIOPORTAL_API}/generic-assay-data/{profile}",
        n_records=n_records,
        notes=notes,
    )
    try:
        with open(cache_path, "w") as fh:
            json.dump(
                {
                    "source": panel.source,
                    "profile": panel.profile,
                    "probes": panel.probes,
                    "n_records": panel.n_records,
                    "betas": panel.betas,
                    "notes": panel.notes,
                },
                fh,
                indent=1,
                sort_keys=True,
            )
    except OSError:
        pass
    return panel


# ---------------------------------------------------------------------------
# Published binary calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MgmtCallTable:
    """Published MGMT promoter calls, one column per source study."""

    sources: Tuple[Tuple[str, str], ...]
    calls: Dict[str, Dict[str, str]]
    vocabulary: Dict[str, Dict[str, int]]
    note: str = ""

    @property
    def patient_ids(self) -> List[str]:
        return sorted(self.calls)

    @property
    def n_patients(self) -> int:
        return len(self.calls)

    def value(self, patient: str, source: Optional[Tuple[str, str]] = None) -> Optional[str]:
        """The call for a patient, optionally restricted to one source study."""
        row = self.calls.get(patient)
        if not row:
            return None
        if source is not None:
            return row.get(f"{source[0]}:{source[1]}")
        for key, value in sorted(row.items()):
            if value:
                return value
        return None

    def binary(self, patient: str) -> Optional[int]:
        """1 for methylated, 0 for unmethylated, None when the vocabulary is neither.

        `Unknown`, `Not Reported` and an empty value are missing. Coercing them
        to 0 would read a missing assay as an unmethylated tumour.
        """
        value = self.value(patient)
        if value is None:
            return None
        token = value.strip().lower()
        if token in METHYLATED_TOKENS:
            return 1
        if token in UNMETHYLATED_TOKENS:
            return 0
        return None

    def binaries(self, patients: Iterable[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for pid in patients:
            b = self.binary(pid)
            if b is not None:
                out[pid] = b
        return out

    def __str__(self) -> str:
        parts = []
        for study, attr in self.sources:
            vocab = self.vocabulary.get(f"{study}:{attr}", {})
            parts.append(
                f"{study}/{attr}: " + ", ".join(f"{k}={v}" for k, v in sorted(vocab.items()))
            )
        return f"published MGMT calls: {self.n_patients} patient(s); " + "; ".join(parts)


def load_published_mgmt_calls(
    *,
    sources: Sequence[Tuple[str, str]] = MGMT_CALL_SOURCES,
    fetcher: Optional[Callable[[str], object]] = None,
    cache_path: str = MGMT_CALL_CACHE,
    use_cache: bool = True,
) -> MgmtCallTable:
    """Fetch published MGMT promoter calls from the cBioPortal clinical API."""
    if use_cache:
        try:
            with open(cache_path) as fh:
                blob = json.load(fh)
            if [tuple(s) for s in blob.get("sources", [])] == [
                tuple(s) for s in sources
            ]:
                return MgmtCallTable(
                    sources=tuple(tuple(s) for s in blob["sources"]),
                    calls={k: dict(v) for k, v in blob["calls"].items()},
                    vocabulary={k: dict(v) for k, v in blob["vocabulary"].items()},
                    note=blob.get("note", ""),
                )
        except (OSError, ValueError, TypeError, KeyError):
            pass

    fetch = fetcher or _default_fetcher
    calls: Dict[str, Dict[str, str]] = {}
    vocabulary: Dict[str, Dict[str, int]] = {}
    for study, attribute in sources:
        payload = fetch(f"{CBIOPORTAL_API}/studies/{study}/clinical-data?projection=SUMMARY")
        records = payload if isinstance(payload, list) else []
        vocab: Dict[str, int] = {}
        for rec in records:
            if not isinstance(rec, dict) or rec.get("clinicalAttributeId") != attribute:
                continue
            pid = rec.get("patientId")
            value = (rec.get("value") or "").strip()
            if not pid or not value:
                continue
            vocab[value] = vocab.get(value, 0) + 1
            calls.setdefault(str(pid), {})[f"{study}:{attribute}"] = value
        vocabulary[f"{study}:{attribute}"] = vocab

    table = MgmtCallTable(
        sources=tuple(tuple(s) for s in sources),
        calls=calls,
        vocabulary=vocabulary,
        note=(
            "published calls, not recomputed here; a binary call is not a beta, so "
            "it is used as a binary covariate and never as a continuous one"
        ),
    )
    try:
        with open(cache_path, "w") as fh:
            json.dump(
                {
                    "sources": [list(s) for s in table.sources],
                    "calls": table.calls,
                    "vocabulary": table.vocabulary,
                    "note": table.note,
                },
                fh,
                indent=1,
                sort_keys=True,
            )
    except OSError:
        pass
    return table


# ---------------------------------------------------------------------------
# Do the two sources agree?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MgmtConcordance:
    """Recomputed STP27 call against the published call, same patients."""

    n_overlap: int
    n_agree: int
    n_methylated_beta_only: int
    n_methylated_call_only: int
    n_not_callable: int
    cutoff: float

    @property
    def agreement(self) -> float:
        return self.n_agree / self.n_overlap if self.n_overlap else float("nan")

    @property
    def verdict(self) -> str:
        if not self.n_overlap:
            return "no patient carries both a beta and a published call"
        if not np.isfinite(self.agreement):
            return "not estimable"
        return (
            f"{100 * self.agreement:.1f}% agreement on {self.n_overlap} patient(s) "
            f"-- the two sources measure the same promoter, so a low agreement "
            f"would mean the recomputed call is not interchangeable with the "
            f"published one"
        )

    def __str__(self) -> str:
        return (
            f"MGMT call concordance: {self.n_agree}/{self.n_overlap} agree "
            f"(beta-only methylated {self.n_methylated_beta_only}, "
            f"call-only methylated {self.n_methylated_call_only}, "
            f"not callable {self.n_not_callable}) at cutoff {self.cutoff:.3f} -- "
            f"{self.verdict}"
        )


def mgmt_call_concordance(
    panel: MgmtBetaPanel,
    calls: MgmtCallTable,
    *,
    cutoff: float = MGMT_METHYLATION_CUTOFF,
) -> MgmtConcordance:
    """Compare the beta-derived call with the published binary call."""
    n_overlap = n_agree = beta_only = call_only = not_callable = 0
    for pid in panel.patient_ids:
        published = calls.binary(pid)
        beta_call = panel.call(pid, cutoff=cutoff)
        if beta_call == "not callable":
            not_callable += 1
            continue
        if published is None:
            continue
        n_overlap += 1
        derived = 1 if beta_call == "methylated" else 0
        if derived == published:
            n_agree += 1
        elif derived == 1:
            beta_only += 1
        else:
            call_only += 1
    return MgmtConcordance(
        n_overlap=n_overlap,
        n_agree=n_agree,
        n_methylated_beta_only=beta_only,
        n_methylated_call_only=call_only,
        n_not_callable=not_callable,
        cutoff=cutoff,
    )


# ---------------------------------------------------------------------------
# The requested mapping: r0 = Baseline_Resistance * (1 - beta)
# ---------------------------------------------------------------------------

#: The unmethylated baseline. It is defined as the calibrated default rather
#: than chosen: `drug_sensitivity_scale_R` is 0.25 in `DEFAULT_CLONES`, so the
#: resistance the term modulates starts at 0.75. A reader who wants a different
#: baseline must pass it, and the audit prints which one was used.
BASELINE_RESISTANCE = 1.0 - DEFAULT_CLONES.drug_sensitivity_scale_R


def mgmt_resistance(
    beta: float, *, baseline_resistance: float = BASELINE_RESISTANCE
) -> float:
    """``r0 = Baseline_Resistance * (1 - beta)``, the requested mapping.

    `beta` is a promoter methylation fraction in [0, 1]: 0 unmethylated (MGMT
    expressed, tumour repairs alkylator damage, resistant) and 1 fully
    methylated (MGMT silenced, tumour sensitive). A value outside [0, 1] is
    refused rather than clamped, because a clamped beta is a fabricated
    measurement.
    """
    if not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
        raise ValueError("beta must be a finite number")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"beta must be a methylation fraction in [0, 1], got {beta}")
    if not 0.0 <= baseline_resistance <= 1.0:
        raise ValueError("baseline_resistance must be in [0, 1]")
    return float(baseline_resistance) * (1.0 - float(beta))


def clone_params_for_mgmt_beta(
    beta: float,
    *,
    base: CloneParams = DEFAULT_CLONES,
    baseline_resistance: float = BASELINE_RESISTANCE,
) -> CloneParams:
    """Bridge the mapped resistance into the engine's clone parameters.

    The engine expresses resistance as a *sensitivity* scale on the resistant
    clone's drug-A kill rate, so ``drug_sensitivity_scale_R = 1 - r0``. With the
    default baseline this reproduces `DEFAULT_CLONES` exactly at beta = 0, which
    is what makes the mapping a re-parameterisation of the calibrated model
    rather than a second, competing one.

    Everything else is carried through unchanged -- including
    ``growth_R = growth_S * (1 - cost_of_resistance)``. That is deliberate but
    it is a real inconsistency: a methylated tumour keeps paying the full
    metabolic cost of a resistance it has lost. `resistance_audit` reports the
    coupling; it is not silently corrected here.
    """
    r0 = mgmt_resistance(beta, baseline_resistance=baseline_resistance)
    return replace(base, drug_sensitivity_scale_R=1.0 - r0)


@dataclass(frozen=True)
class ResistanceAudit:
    """How far the mapping can actually move the engine's drug-A resistance."""

    baseline_resistance: float
    scale_component: float
    kill_ceiling_ratio: float
    total_resistance_unmethylated: float
    total_resistance_fully_methylated: float

    @property
    def reachable_span(self) -> float:
        return self.total_resistance_unmethylated - self.total_resistance_fully_methylated

    @property
    def ceiling_share(self) -> float:
        """Share of the total resistance that the mapped term cannot touch."""
        if self.total_resistance_unmethylated <= 0:
            return float("nan")
        return (
            self.total_resistance_unmethylated - self.reachable_span
        ) / self.total_resistance_unmethylated

    @property
    def verdict(self) -> str:
        return (
            f"the mapped term moves total drug-A resistance from "
            f"{self.total_resistance_unmethylated:.4f} to "
            f"{self.total_resistance_fully_methylated:.4f} -- a span of "
            f"{self.reachable_span:.4f}. The remaining "
            f"{100 * self.ceiling_share:.1f}% of the resistance lives in the "
            f"clone's kill ceiling (kill_max_R/kill_max_S = "
            f"{self.kill_ceiling_ratio:.4f}), which this mapping does not touch"
        )

    def __str__(self) -> str:
        return (
            f"resistance audit: baseline_resistance={self.baseline_resistance:.4f} "
            f"(scale component {self.scale_component:.4f}, kill ceiling ratio "
            f"{self.kill_ceiling_ratio:.4f}) -- {self.verdict}"
        )


def resistance_audit(
    *,
    clones: CloneParams = DEFAULT_CLONES,
    drugs=None,
    baseline_resistance: float = BASELINE_RESISTANCE,
) -> ResistanceAudit:
    """Split the engine's drug-A resistance into the part the term moves and the rest."""
    drugs = drugs if drugs is not None else DEFAULT_DRUGS
    kill_s = float(drugs.kill_max_S[0])
    kill_r = float(drugs.kill_max_R[0])
    if kill_s <= 0:
        raise ValueError("drug A has no sensitive kill rate: resistance is undefined")
    ratio = kill_r / kill_s
    unmethylated = 1.0 - (kill_r * clones.drug_sensitivity_scale_R) / kill_s
    methylated = 1.0 - kill_r / kill_s  # scale = 1 at beta = 1
    return ResistanceAudit(
        baseline_resistance=float(baseline_resistance),
        scale_component=float(baseline_resistance),
        kill_ceiling_ratio=ratio,
        total_resistance_unmethylated=float(unmethylated),
        total_resistance_fully_methylated=float(methylated),
    )


# ---------------------------------------------------------------------------
# What the mapping does to the simulated dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaArmOutcome:
    beta: float
    scale_R: float
    median_pfs_days: float
    pfs_iqr: Tuple[float, float]
    frac_progressed: float
    median_dominance_day: Optional[float]


@dataclass(frozen=True)
class MgmtResistanceSweep:
    """Engine response to real betas, per policy."""

    betas: Tuple[float, ...]
    beta_source: str
    n_replicates: int
    days: int
    geometry: str
    by_policy: Dict[str, Tuple[BetaArmOutcome, ...]]

    def outcomes(self, policy: str) -> Tuple[BetaArmOutcome, ...]:
        return self.by_policy.get(policy, ())

    def span(self, policy: str) -> Tuple[float, float]:
        outs = self.outcomes(policy)
        if not outs:
            return float("nan"), float("nan")
        vals = [o.median_pfs_days for o in outs]
        return min(vals), max(vals)

    @property
    def verdict(self) -> str:
        if not self.by_policy:
            return "not measured"
        parts = []
        for policy, outs in sorted(self.by_policy.items()):
            lo, hi = self.span(policy)
            if not np.isfinite(lo):
                continue
            parts.append(f"{policy}: median PFS {lo:.0f}-{hi:.0f}d over the beta range")
        return (
            f"simulated, not clinical: {self.n_replicates} replicate(s) per beta over "
            f"{self.days} days on {self.geometry}. " + "; ".join(parts)
        )

    def __str__(self) -> str:
        lines = [
            f"MGMT resistance sweep: beta values {[round(b, 4) for b in self.betas]} "
            f"from {self.beta_source}, {self.n_replicates} replicate(s), {self.days} days"
        ]
        for policy, outs in sorted(self.by_policy.items()):
            for o in outs:
                dom = f"{o.median_dominance_day:.0f}" if o.median_dominance_day else "none"
                lines.append(
                    f"  {policy:28s} beta={o.beta:.4f} scale_R={o.scale_R:.4f} "
                    f"median PFS={o.median_pfs_days:6.1f}d IQR=[{o.pfs_iqr[0]:.0f},"
                    f"{o.pfs_iqr[1]:.0f}] progressed={100 * o.frac_progressed:.0f}% "
                    f"dominance day={dom}"
                )
        lines.append(f"  -- {self.verdict}")
        return "\n".join(lines)


def mgmt_resistance_sweep(
    betas: Sequence[float],
    *,
    beta_source: str,
    geometry: Optional[TumorGeometry] = None,
    geometry_size: int = 20,
    policies: Optional[Sequence[object]] = None,
    days: int = 365,
    n_replicates: int = 3,
    initial_resistant_fraction: float = 0.01,
    seed0: int = 0,
) -> MgmtResistanceSweep:
    """Run the calibrated arms at each beta and report what actually changes.

    Each beta is a whole run set with its own clone parameters, so the
    comparison is between two models that differ only in the mapped resistance.
    That is the only way the term's effect is attributable to the term.
    """
    from danon.evosteer import AdaptiveDosePolicy, CollateralSwitchPolicy, MTDPolicy

    geom = geometry if geometry is not None else make_gbm_phantom(geometry_size)
    arms = list(policies) if policies is not None else [
        MTDPolicy(),
        AdaptiveDosePolicy(),
        CollateralSwitchPolicy(),
    ]
    by_policy: Dict[str, List[BetaArmOutcome]] = {}
    for beta in betas:
        clones = clone_params_for_mgmt_beta(beta)
        sim = TumorEvolutionSimulator(geom, clones=clones)
        for pol in arms:
            name = pol.describe()
            pfs: List[float] = []
            doms: List[float] = []
            for rep in range(n_replicates):
                traj = sim.run(
                    pol,
                    days=days,
                    initial_resistant_fraction=initial_resistant_fraction,
                    seed=seed0 + rep,
                )
                pfs.append(traj.pfs_value(days))
                if traj.resistant_dominance_day is not None:
                    doms.append(float(traj.resistant_dominance_day))
            arr = np.asarray(pfs, dtype=float)
            by_policy.setdefault(name, []).append(
                BetaArmOutcome(
                    beta=float(beta),
                    scale_R=float(clones.drug_sensitivity_scale_R),
                    median_pfs_days=float(np.median(arr)),
                    pfs_iqr=(
                        float(np.percentile(arr, 25)),
                        float(np.percentile(arr, 75)),
                    ),
                    frac_progressed=float(np.mean(arr < days)),
                    median_dominance_day=float(np.median(doms)) if doms else None,
                )
            )
    return MgmtResistanceSweep(
        betas=tuple(float(b) for b in betas),
        beta_source=beta_source,
        n_replicates=n_replicates,
        days=days,
        geometry=f"GBM phantom size {geometry_size}",
        by_policy={k: tuple(v) for k, v in by_policy.items()},
    )


# ---------------------------------------------------------------------------
# Honesty envelope
# ---------------------------------------------------------------------------


def mgmt_provenance() -> Dict[str, EvidenceStatus]:
    """What each quantity here is, in the repository's vocabulary."""
    return {
        "mgmt_probe_selection": EvidenceStatus.UNKNOWN,       # literature-derived
        "mgmt_methylation_cutoff": EvidenceStatus.UNKNOWN,    # model-specific
        "mgmt_beta_values": EvidenceStatus.ESTABLISHED,       # read from published per-probe data
        "mgmt_published_calls": EvidenceStatus.ESTABLISHED,   # read from published clinical data
        "mgmt_call_concordance": EvidenceStatus.ESTABLISHED,  # computed, not asserted
        "mgmt_resistance_mapping": EvidenceStatus.UNKNOWN,    # a declared convention
        "mgmt_resistance_effect": EvidenceStatus.ESTABLISHED,  # simulated, per beta
    }


def what_mgmt_cannot_show() -> List[str]:
    """Stated plainly so no number here is read as a clinical result."""
    return [
        "that MGMT methylation changes any patient's outcome: it is a prognostic "
        "marker, and measuring its association in a cohort is not a mechanism",
        "that the mapped resistance is the real resistance: the mapping is a "
        "declared convention anchored on the calibrated default, not a fit",
        "that the simulated PFS differences are treatment effects: the engine's "
        "dynamics class is anchored to a prostate adaptive-therapy trial, and no "
        "GBM outcome is reproduced here",
        "that a beta from a donor tumour equals the patient's own: the panel is a "
        "cohort distribution, and the engine consumes one beta per simulated case",
    ]


def render_mgmt_report(
    *,
    panel: Optional[MgmtBetaPanel] = None,
    calls: Optional[MgmtCallTable] = None,
    sweep: Optional[MgmtResistanceSweep] = None,
    audit: Optional[ResistanceAudit] = None,
) -> str:
    """The text block for the engineering logbook."""
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append("MGMT PROMOTER METHYLATION -> RESISTANCE TERM, MEASURED")
    lines.append("=" * 74)
    lines.append("")
    lines.append("-- the requested mapping --")
    lines.append("  r0 = Baseline_Resistance * (1 - beta)")
    lines.append(
        f"  Baseline_Resistance = 1 - drug_sensitivity_scale_R("
        f"DEFAULT_CLONES) = {BASELINE_RESISTANCE:.4f}"
    )
    lines.append(f"  {audit if audit is not None else resistance_audit()}")
    lines.append("")
    if panel is not None:
        lines.append(f"-- {panel}")
        for note in panel.notes:
            lines.append(f"     note: {note}")
        qs = panel.quantiles()
        lines.append(f"     beta quantiles (0, .25, .5, .75, 1): {[round(q, 4) for q in qs]}")
    else:
        lines.append("-- beta panel not loaded in this run")
    lines.append("")
    if calls is not None:
        lines.append(f"-- {calls}")
        if panel is not None:
            lines.append(f"-- {mgmt_call_concordance(panel, calls)}")
    else:
        lines.append("-- published calls not loaded in this run")
    lines.append("")
    if sweep is not None:
        lines.append(str(sweep))
    else:
        lines.append("-- engine sweep not run in this run")
    lines.append("")
    lines.append("What this cannot show:")
    for item in what_mgmt_cannot_show():
        lines.append(f"  - {item}")
    return "\n".join(lines)


__all__ = [
    "BASELINE_RESISTANCE",
    "CBIOPORTAL_API",
    "METHYLATED_TOKENS",
    "MGMT_BETA_CACHE",
    "MGMT_BETA_PROFILE",
    "MGMT_BETA_STUDY",
    "MGMT_CALL_CACHE",
    "MGMT_CALL_SOURCES",
    "UNMETHYLATED_TOKENS",
    "BetaArmOutcome",
    "MgmtBetaPanel",
    "MgmtCallTable",
    "MgmtConcordance",
    "MgmtResistanceSweep",
    "ResistanceAudit",
    "clone_params_for_mgmt_beta",
    "load_mgmt_beta_panel",
    "load_published_mgmt_calls",
    "mgmt_call_concordance",
    "mgmt_provenance",
    "mgmt_resistance",
    "mgmt_resistance_sweep",
    "render_mgmt_report",
    "resistance_audit",
    "what_mgmt_cannot_show",
]
