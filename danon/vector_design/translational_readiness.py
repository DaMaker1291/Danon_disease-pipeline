"""
TRANSLATIONAL READINESS GATEWAY ENGINE
=======================================
Bridges the gap between in-silico sequence architecture and a physical medicine
sitting in a clinical freezer. This module is a *regulatory gatekeeper*: it does
NOT validate biology — it tracks which physical wet-lab validation assets are
still missing, and refuses to flag a construct as clinically eligible until
explicit verification tokens (lab assays) are supplied.

Gates follow the real MHRA ILAP / US FDA Preclinical-to-IND pathway:

  Tier 1 — In Vitro Efficacy
  Tier 2 — In Vivo & Biodistribution / Durability
  Tier 3 — GLP Toxicology & Genotoxicity
  Tier 4 — Chemistry, Manufacturing & Controls (CMC)
  Tier 5 — Clinical Trial (multi-centre)

The engine computes a transparent readiness stage and a 0-1 completion score.
By design, a fresh in-silico candidate has NO tokens and is reported as
"Phase 0.5 (In Silico Optimization Validated)" with clinical_trial_eligibility
= False — the honest baseline.

CRITICAL: This platform is a computational simulation. It does not prove
biological safety, toxicity, in-vivo efficacy, or curative potential.
"""
import logging
from typing import Dict, List, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Full clinical-titer target: GENETHON/clinical AAV9 processes routinely cite
# >= 1e13 vg/L in suspension bioreactors as a manufacturable floor.
GMP_VG_LITER_FLOOR = 1e13


# --------------------------------------------------------------------------- #
# Wet-lab milestone tokens (all default False / 0 — nothing is assumed)
# --------------------------------------------------------------------------- #
class PreclinicalValidationMilestones(BaseModel):
    # ---- Tier 1: In Vitro Efficacy Assets ----
    ipsc_transduction_verified: bool = Field(
        default=False,
        description="Western blot / flow confirmed LAMP2B expression in human patient iPSC-cardiomyocytes.")
    glycogen_clearance_observed: bool = Field(
        default=False,
        description="Transmission EM shows clearance of cardiomyocyte autophagic vacuoles / glycogen stores.")

    # ---- Tier 2: In Vivo & Biodistribution / Durability ----
    large_animal_pk_validated: bool = Field(
        default=False,
        description="NHP or porcine biodistribution confirming cardiac-to-hepatic ratio matches model weights.")
    durability_months: int = Field(
        default=0, ge=0, le=600,
        description="Months of verified continuous expression in vivo without transgene silencing.")

    # ---- Tier 3: GLP Toxicology & Genotoxicity ----
    glp_tox_cleared: bool = Field(
        default=False,
        description="GLP repeat-dose toxicology (hepatic, cardiac, immunologic) cleared with acceptable NOAEL.")
    genotoxicity_validated: bool = Field(
        default=False,
        description="Insertional/genotoxicity screen (AAV integration hotspot + off-target) cleared.")

    # ---- Tier 4: Chemistry, Manufacturing & Controls (CMC) ----
    gmp_vector_yield_per_liter: float = Field(
        default=0.0, ge=0.0,
        description="Verified viral genome (vg) yield per litre of HEK293 / suspension bioreactor culture.")
    empty_capsid_separation_efficiency: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Purity percentage achieved via ultracentrifugation / chromatography full/empty sorting.")

    # ---- Tier 5: Clinical ----
    multicenter_trial_initiated: bool = Field(
        default=False,
        description="Multi-centre Phase I/II human trial approved and dosed at least one cohort.")


# --------------------------------------------------------------------------- #
# Readiness result models
# --------------------------------------------------------------------------- #
class ReadinessGate(BaseModel):
    id: int
    tier: str
    name: str
    description: str
    verified: bool
    token_field: str
    pending_label: str = "PENDING PHYSICAL LAB DATA"


class TranslationalReadinessResult(BaseModel):
    current_stage: str
    clinical_trial_eligibility: bool
    translational_completion: float = Field(ge=0.0, le=1.0)
    readiness_score_pct: float
    gates: List[ReadinessGate] = Field(default_factory=list)
    required_next_step: str
    verified_count: int
    total_gates: int


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class TranslationalReadinessEngine:
    """Strict gatekeeper. Computes a transparent readiness tier; never asserts
    clinic-readiness without supplied physical validation tokens."""

    def __init__(self, milestones: PreclinicalValidationMilestones = None,
                 gmp_vg_floor: float = GMP_VG_LITER_FLOOR):
        self.milestones = milestones or PreclinicalValidationMilestones()
        self.gmp_vg_floor = gmp_vg_floor

    def _gate_definitions(self) -> List[ReadinessGate]:
        m = self.milestones
        cls = type(m)
        return [
            ReadinessGate(id=1, tier="Tier 1 · In Vitro", name="iPSC Transduction (LAMP2B)",
                          description=cls.model_fields["ipsc_transduction_verified"].description,
                          verified=m.ipsc_transduction_verified, token_field="ipsc_transduction_verified"),
            ReadinessGate(id=2, tier="Tier 1 · In Vitro", name="Glycogen Clearance (TEM)",
                          description=cls.model_fields["glycogen_clearance_observed"].description,
                          verified=m.glycogen_clearance_observed, token_field="glycogen_clearance_observed"),
            ReadinessGate(id=3, tier="Tier 2 · In Vivo", name="Large-Animal PK / Biodistribution",
                          description=cls.model_fields["large_animal_pk_validated"].description,
                          verified=m.large_animal_pk_validated, token_field="large_animal_pk_validated"),
            ReadinessGate(id=4, tier="Tier 2 · In Vivo", name="Durability (silencing-free)",
                          description="Continuous expression tracked in vivo without transgene silencing.",
                          verified=m.durability_months >= 12, token_field="durability_months",
                          pending_label=f"PENDING PHYSICAL LAB DATA ({m.durability_months} mo)"),
            ReadinessGate(id=5, tier="Tier 3 · Safety", name="GLP Toxicology",
                          description=cls.model_fields["glp_tox_cleared"].description,
                          verified=m.glp_tox_cleared, token_field="glp_tox_cleared"),
            ReadinessGate(id=6, tier="Tier 3 · Safety", name="Genotoxicity / Insertional",
                          description=cls.model_fields["genotoxicity_validated"].description,
                          verified=m.genotoxicity_validated, token_field="genotoxicity_validated"),
            ReadinessGate(id=7, tier="Tier 4 · CMC", name="GMP Vector Yield",
                          description=cls.model_fields["gmp_vector_yield_per_liter"].description,
                          verified=m.gmp_vector_yield_per_liter >= self.gmp_vg_floor,
                          token_field="gmp_vector_yield_per_liter",
                          pending_label=f"PENDING PHYSICAL LAB DATA ({m.gmp_vector_yield_per_liter:.1e} vg/L)"),
            ReadinessGate(id=8, tier="Tier 4 · CMC", name="Full/Empty Capsid Purity",
                          description=cls.model_fields["empty_capsid_separation_efficiency"].description,
                          verified=m.empty_capsid_separation_efficiency >= 90.0,
                          token_field="empty_capsid_separation_efficiency",
                          pending_label=f"PENDING PHYSICAL LAB DATA ({m.empty_capsid_separation_efficiency:.0f}%)"),
            ReadinessGate(id=9, tier="Tier 5 · Clinical", name="Multi-Centre Human Trial",
                          description=cls.model_fields["multicenter_trial_initiated"].description,
                          verified=m.multicenter_trial_initiated, token_field="multicenter_trial_initiated"),
        ]

    def evaluate_translational_gate(self) -> TranslationalReadinessResult:
        gates = self._gate_definitions()
        verified = [g for g in gates if g.verified]
        n = len(gates)
        completion = len(verified) / n

        # Hard clinic-eligibility: every physical-validation gate must carry a token.
        is_clinic_ready = len(verified) == n

        if is_clinic_ready:
            stage = "Phase 4 (Clinical Trial Underway)"
            next_step = "Compile dossier; submit MHRA ILAP / FDA BLA package."
        elif self.milestones.multicenter_trial_initiated:
            stage = "Phase 3 (Clinical)"
            next_step = "Complete enrolment; track primary endpoint (LVMI regression)."
        elif self.milestones.gmp_vector_yield_per_liter >= self.gmp_vg_floor and self.milestones.glp_tox_cleared:
            stage = "Phase 2 (CMC + Tox Locked)"
            next_step = "Initiate multi-centre trial application with MHRA ILAP."
        elif self.milestones.large_animal_pk_validated:
            stage = "Phase 1.5 (In Vivo Validated)"
            next_step = "Execute GLP toxicology + scale GMP vector production."
        elif self.milestones.ipsc_transduction_verified:
            stage = "Phase 1 (In Vitro Validated)"
            next_step = "Initiate large-animal biodistribution & durability studies."
        else:
            stage = "Phase 0.5 (In Silico Optimization Validated)"
            next_step = "Initiate Step 1: in vitro patient-line iPSC-cardiomyocyte transduction assays."

        return TranslationalReadinessResult(
            current_stage=stage,
            clinical_trial_eligibility=is_clinic_ready,
            translational_completion=round(completion, 4),
            readiness_score_pct=round(completion * 100, 1),
            gates=gates,
            required_next_step=next_step,
            verified_count=len(verified),
            total_gates=n,
        )
