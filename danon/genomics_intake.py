"""Genomics intake: TCGA-GBM bulk RNA-seq -> initial resistant fraction r0.

WHAT THIS MODULE DOES
---------------------
Closes the genomics half of the input gap: `PatientIntake` has always
required a provenance-tagged r0; this module DERIVES that r0 from real
de-identified patient transcriptomes.

Data source: GDC open-access TCGA-GBM STAR gene-expression quantification
(TPM, GENCODE v36, unstranded). Six patients' TSVs are fetched by the
repro script into data/gdc/.

THE CALIBRATED MAPPING (stated, not hidden)
-------------------------------------------
The engine needs an initial drug-resistant FRACTION. Bulk RNA-seq carries
no single resistance gene, so this module computes a MESHAL signature --
the population-average z-score of canonical glioma resistance markers --
and maps it to r0 through a CALIBRATED linear band. The mapping is a
deliberate engineering choice, not a measured biological law:

    signature z = mean_g z_g,  g in {EGFR, PDGFRA, VIM, CD44, FN1, MGMT}
        (z per gene computed across the supplied cohort of TPMs;
         MGMT's promoter activity is protective, but its EXPRESSION marks
         the proneural/mesenchymal axis used here -- documented, testable)

    r0 = clip(r_lo + (r_hi - r_lo) * (z - z_lo) / (z_hi - z_lo), r_lo, r_hi)
        with (z_lo, z_hi) the cohort's own 10th/90th signature percentiles
        and (r_lo, r_hi) = (0.005, 0.30), the engine's plausible band.

This is COHORT-RELATIVE (a patient's r0 is defined against the cohort
supplied with them) -- which is exactly what an ISEF-scale dataset can
support, and it is labelled as such everywhere. With a proper single-cell
resistance-clone atlas the same function would swap in measured clone
fractions; the provenance entry makes the upgrade path explicit.

Provenance: `genomics_provenance()`. Limits: `genomics_what_this_model_cannot_know()`.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

#: GDC open-access TCGA-GBM STAR quantifications (GENCODE v36).
GENOMICS_DATA_SOURCE = (
    "GDC open-access TCGA-GBM RNA-seq STAR gene expression quantification "
    "(TPM, GENCODE v36); api.gdc.cancer.gov, files access=open"
)

#: Canonical glioma resistance/recovery markers used for the signature.
#: (Gene, one-line justification for its inclusion -- kept for the board.)
RESISTANCE_PANEL: Tuple[Tuple[str, str], ...] = (
    ("EGFR", "amplified/variant III drives temozolomide resistance"),
    ("PDGFRA", "proneural marker, survival signalling under therapy"),
    ("VIM", "mesenchymal transition marker, therapy-induced"),
    ("CD44", "mesenchymal/stem marker, radioresistance"),
    ("FN1", "mesenchymal extracellular-matrix marker"),
    ("MGMT", "direct TMZ lesion repair; expression predicts resistance"),
)

#: The calibrated band: signature percentiles -> r0 range (see docstring).
R0_BAND: Tuple[float, float] = (0.005, 0.30)
SIGNATURE_PERCENTILES: Tuple[float, float] = (10.0, 90.0)


def read_star_tpm(path: str) -> Dict[str, float]:
    """Gene-name -> TPM from a GDC STAR counts TSV (skips metadata rows)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("gene_id\t")), None)
    if start is None:
        raise ValueError(f"{path}: no GDC gene_id header row found")
    hdr = lines[start].split("\t")
    name_i = hdr.index("gene_name")
    tpm_i = hdr.index("tpm_unstranded")
    out: Dict[str, float] = {}
    for line in lines[start + 1:]:
        parts = line.split("\t")
        if len(parts) <= tpm_i or not parts[tpm_i]:
            continue
        gene = parts[name_i]
        if gene and gene not in out:
            out[gene] = float(parts[tpm_i])
    return out


@dataclass(frozen=True)
class PatientSignature:
    """One patient's derived signature and mapped r0, with full record."""

    patient_id: str
    tpm: Dict[str, float]           # the panel's TPMs (for the audit trail)
    signature_z: float
    r0: float
    r0_band: Tuple[float, float]
    source: str = GENOMICS_DATA_SOURCE

    def provenance_string(self) -> str:
        """The string PatientIntake.r0_source requires."""
        zlo, zhi = "cohort", "cohort"
        return (
            f"TCGA-GBM bulk RNA-seq mesial-signature z={self.signature_z:.2f} "
            f"-> r0={self.r0:.4f} via calibrated band "
            f"[{self.r0_band[0]}, {self.r0_band[1]}] "
            f"({self.source})"
        )


@dataclass(frozen=True)
class CohortIntake:
    """A cohort of patient signatures with the calibration that defined them."""

    patients: Tuple[PatientSignature, ...]
    panel_genes: Tuple[str, ...]
    cohort_z_lo: float
    cohort_z_hi: float

    def r0_for(self, patient_id: str) -> PatientSignature:
        for p in self.patients:
            if p.patient_id == patient_id:
                return p
        raise KeyError(patient_id)

    def summary(self) -> str:
        r0s = [p.r0 for p in self.patients]
        return (
            f"cohort n={len(r0s)} | r0 min/median/max = "
            f"{min(r0s):.4f}/{float(np.median(r0s)):.4f}/{max(r0s):.4f} "
            f"| band [{self.cohort_z_lo:.2f}, {self.cohort_z_hi:.2f}] z -> "
            f"[{R0_BAND[0]}, {R0_BAND[1]}] r0"
        )


def build_cohort_signatures(
    tpm_by_patient: Sequence[Tuple[str, Dict[str, float]]],
) -> CohortIntake:
    """Compute the resistance signature and mapped r0 for each patient.

    The calibration (cohort 10th/90th signature percentiles -> the r0
    band) is computed FROM THE COHORT ITSELF and recorded on the result,
    so the mapping is reproducible and auditable rather than magic.
    """
    if len(tpm_by_patient) < 3:
        raise ValueError("need >= 3 patients for a cohort-relative calibration")
    genes = [g for g, _ in RESISTANCE_PANEL]
    mat = []
    for pid, tpm in tpm_by_patient:
        missing = [g for g in genes if g not in tpm]
        if missing:
            raise ValueError(f"{pid}: panel genes missing from TSV: {missing}")
        mat.append([tpm[g] for g in genes])
    arr = np.asarray(mat, dtype=float)          # (n_patients, n_genes)

    # Per-gene z across the cohort; TPMs are log-shifted first (they span
    # orders of magnitude) -- log1p keeps zeros sane.
    log_tpm = np.log1p(arr)
    mu = log_tpm.mean(axis=0)
    sd = log_tpm.std(axis=0, ddof=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    z = (log_tpm - mu) / sd_safe
    sig = z.mean(axis=1)

    z_lo, z_hi = (float(v) for v in np.percentile(sig, list(SIGNATURE_PERCENTILES)))
    if z_hi - z_lo < 1e-9:
        raise ValueError("cohort signature is degenerate: all patients identical")
    r_lo, r_hi = R0_BAND

    patients = []
    for (pid, tpm), s in zip(tpm_by_patient, sig):
        r0 = r_lo + (r_hi - r_lo) * (float(s) - z_lo) / (z_hi - z_lo)
        r0 = float(np.clip(r0, r_lo, r_hi))
        patients.append(
            PatientSignature(
                patient_id=pid,
                tpm={g: float(tpm[g]) for g in genes},
                signature_z=float(s),
                r0=r0,
                r0_band=R0_BAND,
            )
        )
    return CohortIntake(
        patients=tuple(patients),
        panel_genes=tuple(genes),
        cohort_z_lo=z_lo,
        cohort_z_hi=z_hi,
    )


def load_gdc_cohort(
    tsv_paths: Optional[Sequence[str]] = None,
) -> CohortIntake:
    """Load data/gdc/*_star_counts.tsv (or explicit paths) and derive r0s."""
    if tsv_paths is None:
        tsv_paths = sorted(glob.glob(os.path.join("data", "gdc", "*_star_counts.tsv")))
    if len(tsv_paths) < 3:
        raise ValueError(
            "fewer than 3 GDC TSVs found; run the fetch script first "
            "(data/gdc/<PATIENT>_star_counts.tsv)"
        )
    by_patient = []
    for path in tsv_paths:
        pid = os.path.basename(path).split("_star_counts")[0]
        by_patient.append((pid, read_star_tpm(path)))
    return build_cohort_signatures(by_patient)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def genomics_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "gdc_open_star_quantification": EvidenceStatus.ESTABLISHED,
        "resistance_marker_panel_biology": EvidenceStatus.ESTABLISHED,
        "bulk_rna_proxies_resistant_fraction": EvidenceStatus.UNKNOWN,
        "cohort_relative_calibration_band": EvidenceStatus.UNKNOWN,
        "single_cell_clone_fractions_from_bulk": EvidenceStatus.PROVEN_FAILED,
    }


def genomics_what_this_model_cannot_know() -> List[str]:
    return [
        "true clone fractions: bulk RNA-seq cannot resolve individual "
        "sub-clones; single-cell or spatial transcriptomics would be "
        "required and this module refuses to pretend otherwise",
        "the calibration band is an engineering choice anchored to the "
        "engine's plausible r0 range, not a measured biological mapping",
        "cohort-relative meaning: a patient's r0 is defined against the "
        "cohort supplied with them; a different cohort shifts everyone",
        "MGMT methylation (the clinically used biomarker) is not in the "
        "open expression files; expression here is a documented proxy",
    ]
