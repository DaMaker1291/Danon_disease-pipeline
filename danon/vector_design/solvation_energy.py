"""
PHASE 20 — 3D Epitope Solvation Free-Energy Profile
===================================================
Computes the change in solvation free energy (ΔΔG_solv, kcal/mol) across the
VR-IV and VR-VIII loops after electrostatic charge-masking, ensuring the capsid
stays soluble in human plasma and does not aggregate.

ΔG_solv per residue = ΔG_nonpolar + ΔG_electrostatic(Born)
  ΔG_nonpolar        = σ · SASA           (atomic solvation parameter × exposed area)
  ΔG_electrostatic   = -0.5 · 332 · q² / r · (1 - 1/ε_solvent)   (Born self-energy)

ΔΔG_solv = Σ [ΔG_solv(mutant) - ΔG_solv(wild-type)] over mutated residues.
A positive ΔΔG_solv destabilises solvation (aggregation risk); the phase gates on
a configurable maximum destabilisation bound.

References:
  Eisenberg & McLachlan 1986 (solvation free energy), Nature 319:199.
  Born 1920 (ion solvation); Sitkoff, Sharp & Honig 1994 (Born in continuum).
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EPS_SOLVENT = 78.5
COULOMB_FACTOR = 332.0  # kcal·mol⁻¹·Å·e⁻²

# Fauchère-Pliska octanol-water transfer free energies (kcal/mol); positive =
# hydrophobic (prefers to bury), negative = hydrophilic (prefers solvent).
AA_TRANSFER_DG = {"A": 0.31, "R": -1.01, "N": -0.60, "D": -0.77, "C": 1.54,
                  "Q": -0.22, "E": -0.64, "G": 0.0, "H": 0.13, "I": 1.80,
                  "L": 1.70, "K": -0.99, "M": 1.23, "F": 1.79, "P": 0.72,
                  "S": -0.04, "T": 0.26, "W": 2.25, "Y": 0.96, "V": 1.22}

# Maximum side-chain solvent accessible surface area (Å², Tien 2013 theoretical)
AA_MAX_SASA = {"A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
               "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
               "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
               "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0}

AA_CHARGE = {"R": 1.0, "K": 1.0, "H": 0.5, "D": -1.0, "E": -1.0}
AA_RADIUS = {"A": 1.80, "C": 2.10, "D": 2.40, "E": 2.60, "F": 3.00, "G": 1.50,
             "H": 2.70, "I": 2.90, "K": 3.10, "L": 2.90, "M": 3.00, "N": 2.50,
             "P": 2.20, "Q": 2.60, "R": 3.20, "S": 2.00, "T": 2.20, "V": 2.70,
             "W": 3.40, "Y": 3.20}

# Empirical atomic solvation parameter (kcal·mol⁻¹·Å⁻²)
SIGMA_NONPOLAR = 0.0072


class ResidueSolvation(BaseModel):
    position: int
    aa: str
    accessibility: float
    dg_nonpolar: float
    dg_electrostatic: float
    dg_solv_total: float


class RegionSolvation(BaseModel):
    region: str
    dg_solv_wild_type: float
    dg_solv_mutant: float
    ddg_solv: float
    residue_profiles: List[ResidueSolvation] = Field(default_factory=list)


class SolvationResult(BaseModel):
    regions: List[RegionSolvation] = Field(default_factory=list)
    ddg_solv_total: float
    aggregation_risk_score: float = Field(ge=0.0, le=1.0)
    plasma_soluble: bool
    max_allowed_ddg: float


class SolvationEnergyEngine:
    """ΔΔG_solv profiler over the antigenic variable-region loops."""

    def __init__(self, max_allowed_ddg: float = 2.5):
        self.max_allowed_ddg = max_allowed_ddg

    def _born_energy(self, charge: float, radius: float) -> float:
        if radius < 1e-6 or charge == 0.0:
            return 0.0
        return -0.5 * COULOMB_FACTOR * charge ** 2 / radius * (1.0 - 1.0 / EPS_SOLVENT)

    def residue_solvation(self, aa: str, accessibility: float) -> ResidueSolvation:
        sasa = AA_MAX_SASA.get(aa, 180.0) * accessibility
        # nonpolar term scaled by transfer free energy sign/magnitude
        dg_np = SIGMA_NONPOLAR * sasa * AA_TRANSFER_DG.get(aa, 0.0)
        charge = AA_CHARGE.get(aa, 0.0)
        dg_es = self._born_energy(charge, AA_RADIUS.get(aa, 2.0)) * accessibility
        return ResidueSolvation(
            position=0, aa=aa, accessibility=round(accessibility, 3),
            dg_nonpolar=round(dg_np, 4), dg_electrostatic=round(dg_es, 4),
            dg_solv_total=round(dg_np + dg_es, 4),
        )

    def profile_region(self, region: str, wild_type: str, mutant: str,
                       coords: Dict[int, Tuple[float, float, float, float]]) -> RegionSolvation:
        profiles: List[ResidueSolvation] = []
        dg_wt = 0.0
        dg_mut = 0.0
        for pos, (_, _, _, access) in coords.items():
            idx = pos - 1
            if idx >= len(mutant) or idx >= len(wild_type):
                continue
            wt_res = self.residue_solvation(wild_type[idx], access)
            mut_res = self.residue_solvation(mutant[idx], access)
            dg_wt += wt_res.dg_solv_total
            dg_mut += mut_res.dg_solv_total
            mut_res.position = pos
            profiles.append(mut_res)
        return RegionSolvation(
            region=region,
            dg_solv_wild_type=round(dg_wt, 4),
            dg_solv_mutant=round(dg_mut, 4),
            ddg_solv=round(dg_mut - dg_wt, 4),
            residue_profiles=profiles,
        )

    def evaluate(self, wild_type: str, mutant: str,
                 region_coords: Dict[str, Dict]) -> SolvationResult:
        regions: List[RegionSolvation] = []
        total = 0.0
        for region, coords in region_coords.items():
            rs = self.profile_region(region, wild_type, mutant, coords)
            regions.append(rs)
            total += rs.ddg_solv
        # Positive ΔΔG_solv = less favourable solvation = aggregation risk.
        # Negative ΔΔG_solv means the masked surface is MORE soluble (desirable).
        risk = float(np.clip(max(0.0, total) / (self.max_allowed_ddg * 2.0), 0.0, 1.0))
        return SolvationResult(
            regions=regions,
            ddg_solv_total=round(total, 4),
            aggregation_risk_score=round(risk, 4),
            plasma_soluble=total <= self.max_allowed_ddg,
            max_allowed_ddg=self.max_allowed_ddg,
        )
