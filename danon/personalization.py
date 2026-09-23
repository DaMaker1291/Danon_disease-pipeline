"""
Patient personalization layer for ischemic heart disease (IHD).

This is the orchestration layer that turns a *blood panel* into a
*patient-specific manufacturing order*. It does not invent new biology; it
wires primitives that already exist in Danon:

    StoichiometricCalculator.optimize_ratio   -> personalized empty:full decoy ratio
    HLADecoupler.evaluate                     -> epitope burden of the payload
    NAbAssaySimulator.simulate_well           -> neutralization escape of the capsid
    DualVectorMOIOptimizer                    -> dose / MOI for dual-vector payloads

SCIENTIFIC SCOPE — read this before believing any output.

Ischemic heart disease is *polygenic and environmental*. Unlike Danon disease
(a single LAMP2 defect), there is no one broken part in the blood that can be
read out and replaced. Gene therapy for IHD delivers a *therapeutic transgene*
(angiogenic, calcium-handling, or lipid-lowering), it does not repair a germline
defect.

Therefore the genuinely personalizable axis is **immunological and
pharmacokinetic**, not "design a new protein from the patient's blood":

  1. Pre-existing anti-AAV9 neutralizing antibodies (NAb titer) -> decoy ratio
  2. HLA haplotype -> which payload epitopes will be presented
  3. Complement status -> shielding / immune-suppression requirements
  4. Body mass, cardiac function -> dose

Everything here is a computational prediction. It is not a cure, not a
validated therapy, and not clinically actionable. Any real use requires
wet-lab and clinical validation under regulatory supervision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from danon.stoichiometric_calc import StoichiometricCalculator, DecoyOptimizationResult
from danon.hla_decoupler import HLADecoupler, HLADecouplerResult
from danon.nab_assay_simulator import NAbAssaySimulator, NAbAssayWell
from danon.dual_vector_moi_optimizer import DualVectorMOIOptimizer

logger = logging.getLogger(__name__)

# The repo's own clinical rule: titers at or above this are an exclusion
# criterion in most trials.
NAB_EXCLUSION_TITER = 1000.0


# ---------------------------------------------------------------------------
# Blood-panel intake
# ---------------------------------------------------------------------------

@dataclass
class PatientBloodPanel:
    """Everything the pipeline can actually read from a blood draw + chart.

    Plasma -> serology.  PBMC -> HLA typing.  Serum -> lipids / cardiac markers.
    """

    patient_id: str
    age_years: int
    weight_kg: float

    # --- plasma serology -------------------------------------------------
    nab_titer: float = 5.0              # anti-AAV9 neutralizing antibody titer
    anti_peg_titer: float = 0.0         # relevant if an LNP modality is chosen
    complement_c3a_ng_ml: float = 0.0   # activation marker

    # --- PBMC immunogenetics --------------------------------------------
    hla_alleles: Tuple[str, ...] = ()   # e.g. ("HLA-A*02:01", "HLA-DRB1*15:01")

    # --- serum chemistry / cardiac markers ------------------------------
    ldl_mg_dl: float = 100.0
    lipoprotein_a_nmol_l: float = 0.0
    troponin_ng_l: float = 0.0
    bnp_pg_ml: float = 0.0
    crp_mg_l: float = 0.0

    # --- genotype flags (WGS/WES panel) ---------------------------------
    # e.g. ("LDLR_pathogenic", "PCSK9_gof", "APOB_pathogenic")
    genotype_flags: Tuple[str, ...] = ()

    # --- cardiac function ------------------------------------------------
    lvef_pct: float = 50.0              # left-ventricular ejection fraction


class PersonalizationTier(str, Enum):
    """How much bespoke work this patient actually needs."""

    OFF_THE_SHELF = "off_the_shelf"        # validated vector, personalized dose only
    HLA_PERSONALIZED = "hla_personalized"  # epitope-edited payload for this haplotype
    BESPOKE = "bespoke"                    # dual-vector / uncommon transgene
    NOT_ELIGIBLE = "not_eligible"          # immune barrier exceeds safe envelope


class IschemicIndication(str, Enum):
    STABLE_ANGINA = "stable_angina"
    POST_MI = "post_myocardial_infarction"
    CHRONIC_ISCHEMIC_CARDIOMYOPATHY = "chronic_ischemic_cardiomyopathy"
    FAMILIAL_HYPERCHOLESTEROLEMIA = "familial_hypercholesterolemia"


class Transgene(str, Enum):
    """Therapeutic payload classes actually explored for cardiac ischemia."""

    VEGF_A165 = "VEGF-A165"        # therapeutic angiogenesis
    SDF1A = "SDF-1alpha"           # progenitor cell homing
    SERCA2A = "SERCA2a"            # calcium handling (CUPID-class approach)
    PCSK9_KD = "PCSK9_knockdown"   # lipid lowering
    LDLR = "LDLR"                  # familial hypercholesterolemia rescue


class Modality(str, Enum):
    AAV9 = "AAV9"                  # durable, single dose, immunogenic
    LNP_MRNA = "LNP-mRNA"          # redosable, transient, PEG-immunogenic


# ---------------------------------------------------------------------------
# Transgene catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransgeneSpec:
    """A candidate payload and the peptide segments used for epitope screening.

    NOTE: the `epitope_segments` here are short *representative* windows used to
    exercise the HLA screen. Real use requires the full translated sequence and
    a properly validated epitope predictor with the patient's actual alleles.
    """

    transgene: Transgene
    indications: Tuple[IschemicIndication, ...]
    modality_preference: Modality
    payload_kb: float                      # coding size, must fit the vector
    epitope_segments: Tuple[str, ...]
    rationale: str


TRANSGENE_CATALOG: Tuple[TransgeneSpec, ...] = (
    TransgeneSpec(
        transgene=Transgene.VEGF_A165,
        indications=(
            IschemicIndication.STABLE_ANGINA,
            IschemicIndication.POST_MI,
        ),
        modality_preference=Modality.AAV9,
        payload_kb=0.6,
        epitope_segments=("KSWSVYVGAR", "ENGHSSDTGQ"),
        rationale="Therapeutic angiogenesis in chronically hypoperfused myocardium.",
    ),
    TransgeneSpec(
        transgene=Transgene.SDF1A,
        indications=(IschemicIndication.POST_MI,),
        modality_preference=Modality.AAV9,
        payload_kb=0.27,
        epitope_segments=("KPVSLSYRCP", "GWKKNRQFFK"),
        rationale="Progenitor cell homing to peri-infarct border zone.",
    ),
    TransgeneSpec(
        transgene=Transgene.SERCA2A,
        indications=(
            IschemicIndication.CHRONIC_ISCHEMIC_CARDIOMYOPATHY,
            IschemicIndication.POST_MI,
        ),
        modality_preference=Modality.AAV9,
        payload_kb=3.0,
        epitope_segments=("VGEATETALT", "LISLVEVNLQ"),
        rationale="Restores sarcoplasmic-reticulum calcium handling in hibernating myocardium.",
    ),
    TransgeneSpec(
        transgene=Transgene.PCSK9_KD,
        indications=(
            IschemicIndication.STABLE_ANGINA,
            IschemicIndication.CHRONIC_ISCHEMIC_CARDIOMYOPATHY,
        ),
        modality_preference=Modality.LNP_MRNA,
        payload_kb=0.3,
        epitope_segments=("SVKAWPREDS", "GLDGTRTAAT"),
        rationale="Sustained LDL lowering; redosable if transient modality is used.",
    ),
    TransgeneSpec(
        transgene=Transgene.LDLR,
        indications=(IschemicIndication.FAMILIAL_HYPERCHOLESTEROLEMIA,),
        modality_preference=Modality.AAV9,
        payload_kb=2.6,
        epitope_segments=("WTDHLQLNVA", "TGRKGVLSSL"),
        rationale="Restores hepatic LDL clearance in LDLR-pathogenic patients.",
    ),
)


# ---------------------------------------------------------------------------
# Assessment + regimen
# ---------------------------------------------------------------------------

@dataclass
class PatientAssessment:
    """Read-only interpretation of the blood panel."""

    patient_id: str
    titer_classification: str
    nab_titer: float
    complement_risk: float
    hla_epitope_hits: int
    hla_decoupled: bool
    strongest_binder_ic50_nm: float
    immunogenicity_burden: str          # "low" | "moderate" | "high"
    eligible: bool
    dominant_allele: str = ""
    per_allele_hits: Dict[str, int] = field(default_factory=dict)
    deimmunization_targets: List[str] = field(default_factory=list)
    hla_personalized: bool = False
    exclusion_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class PersonalizedRegimen:
    """The patient-specific manufacturing + dosing order."""

    patient_id: str
    tier: PersonalizationTier
    modality: Modality
    transgene: Transgene
    promoter_strategy: str

    dose_vg_per_kg: float
    empty_full_ratio: float
    ratio_range: Tuple[float, float]

    capsid_strategy: str
    epitope_edit_required: bool
    epitope_edit_targets: List[str]

    complement_mitigation: str
    immune_suppression: str

    predicted_free_nab_fraction: float
    predicted_complement_risk: float
    manufacturing_cost_multiplier: float

    rationale: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_manufacturing_order(self) -> Dict:
        """Flat dict for the plasmid builder / LIMS / robot compiler."""
        return {
            "patient_id": self.patient_id,
            "tier": self.tier.value,
            "modality": self.modality.value,
            "transgene": self.transgene.value,
            "promoter_strategy": self.promoter_strategy,
            "capsid_strategy": self.capsid_strategy,
            "dose_vg_per_kg": self.dose_vg_per_kg,
            "empty_full_ratio": round(self.empty_full_ratio, 2),
            "empty_full_ratio_range": [round(r, 2) for r in self.ratio_range],
            "epitope_edit_required": self.epitope_edit_required,
            "epitope_edit_targets": list(self.epitope_edit_targets),
            "complement_mitigation": self.complement_mitigation,
            "immune_suppression": self.immune_suppression,
            "manufacturing_cost_multiplier": round(self.manufacturing_cost_multiplier, 2),
        }

    def render_report(self) -> str:
        """Human-readable summary for a lab notebook or poster."""
        lines = [
            f"PERSONALIZED REGIMEN — {self.patient_id}",
            "=" * 58,
            f"Tier              : {self.tier.value}",
            f"Modality          : {self.modality.value}",
            f"Transgene         : {self.transgene.value}",
            f"Promoter          : {self.promoter_strategy}",
            f"Capsid            : {self.capsid_strategy}",
            f"Dose              : {self.dose_vg_per_kg:.2e} vg/kg",
            f"Empty:Full ratio  : {self.empty_full_ratio:.1f}:1 "
            f"(range {self.ratio_range[0]:.1f}-{self.ratio_range[1]:.1f})",
            f"Epitope editing   : {'REQUIRED' if self.epitope_edit_required else 'not required'}",
            f"Cost multiplier   : {self.manufacturing_cost_multiplier:.2f}x",
            "",
            "Predicted (computational only, not validated):",
            f"  free NAb fraction    : {self.predicted_free_nab_fraction:.3f}",
            f"  complement risk      : {self.predicted_complement_risk:.3f}",
            "",
            "Rationale:",
        ]
        lines += [f"  - {r}" for r in self.rationale]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class PatientPersonalizationEngine:
    """Turns a blood panel into a patient-specific regimen.

    Tiering philosophy: most patients should receive a validated off-the-shelf
    vector with a *personalized dose and decoy ratio*. Only patients whose
    immune profile genuinely breaks the standard product get bespoke work.
    Bespoke manufacturing is weeks of turnaround and orders of magnitude more
    expensive, so it must be reserved, not default.
    """

    def __init__(
        self,
        dose_vg_per_kg: float = 5e13,
        epitope_hit_rate_threshold: float = 0.5,
        complement_risk_threshold: float = 0.5,
        ic50_cutoff_nm: float = 500.0,
    ) -> None:
        self.dose_vg_per_kg = dose_vg_per_kg
        # Burden is judged on the *fraction* of screened windows that present, not
        # an absolute count: payloads vary in length, so a fixed count threshold
        # makes the high-burden tier unreachable for short payloads.
        self.epitope_hit_rate_threshold = epitope_hit_rate_threshold
        self.complement_risk_threshold = complement_risk_threshold
        self._stoich = StoichiometricCalculator()
        self._hla = HLADecoupler(ic50_cutoff_nm=ic50_cutoff_nm)
        self._nab = NAbAssaySimulator()
        self._moi = DualVectorMOIOptimizer()

    # -- step 1: read the blood ------------------------------------------
    def assess(
        self,
        panel: PatientBloodPanel,
        spec: Optional[TransgeneSpec] = None,
    ) -> PatientAssessment:
        """Interpret the panel. If `spec` is given, only that payload's
        epitopes are screened; otherwise the whole catalog is screened."""
        titer_class = self._stoich.classify_titer(panel.nab_titer)
        decoy: DecoyOptimizationResult = self._stoich.optimize_ratio(
            panel.nab_titer, dose_vg_per_kg=self.dose_vg_per_kg, weight_kg=panel.weight_kg
        )
        complement_risk = float(decoy.complement_activation_risk)

        # Flag elevated complement from the direct assay, not just the model.
        if panel.complement_c3a_ng_ml >= 100.0:
            complement_risk = min(1.0, complement_risk + 0.3)

        segments = list(spec.epitope_segments) if spec is not None else list(self._default_segments())
        if panel.hla_alleles:
            # Screen against this patient's actual haplotype rather than the
            # cohort-level DRB1 supertype.
            hla: HLADecouplerResult = HLADecoupler(
                ic50_cutoff_nm=self._hla.ic50_cutoff_nm,
                patient_alleles=tuple(panel.hla_alleles),
            ).evaluate(segments)
        else:
            hla = self._hla.evaluate(segments)

        windows = max(len(segments), 1)
        hit_rate = hla.high_affinity_hits / windows
        if hla.high_affinity_hits == 0:
            burden = "low"
        elif hit_rate >= self.epitope_hit_rate_threshold:
            burden = "high"
        else:
            burden = "moderate"

        exclusions: List[str] = []
        if panel.nab_titer >= NAB_EXCLUSION_TITER:
            exclusions.append(
                f"NAb titer {panel.nab_titer:.0f} >= {NAB_EXCLUSION_TITER:.0f}: "
                "trial exclusion criterion; plasmapheresis or immune suppression required first"
            )
        if panel.lvef_pct < 25.0:
            exclusions.append(
                f"LVEF {panel.lvef_pct:.0f}% below intervention window; too advanced for "
                "a transgene-based approach"
            )
        if panel.age_years < 18:
            exclusions.append("Pediatric patient; outside modeled adult envelope")

        hla_personalized = bool(panel.hla_alleles) and not hla.unresolved_alleles
        deimmunization_targets = [b.core_9mer for b in hla.binders if b.immunogenic]

        notes: List[str] = []
        if hla_personalized:
            notes.append(
                f"Screened against patient haplotype ({len(panel.hla_alleles)} alleles); "
                f"dominant liability = {hla.dominant_allele or 'none'}."
            )
        elif panel.hla_alleles:
            notes.append(
                "Some patient alleles were outside the modelled supertype library "
                f"({', '.join(hla.unresolved_alleles)}); those fell back to the DRB1 "
                "supertype, so the epitope call is only partly patient-specific."
            )
        if panel.anti_peg_titer > 0:
            notes.append(
                "Anti-PEG titer present; if an LNP modality is selected this predicts "
                "accelerated clearance and possible hypersensitivity."
            )

        return PatientAssessment(
            patient_id=panel.patient_id,
            titer_classification=titer_class,
            nab_titer=panel.nab_titer,
            complement_risk=complement_risk,
            hla_epitope_hits=hla.high_affinity_hits,
            hla_decoupled=hla.decoupled,
            strongest_binder_ic50_nm=hla.strongest_binder_ic50_nm,
            immunogenicity_burden=burden,
            eligible=not exclusions,
            dominant_allele=hla.dominant_allele,
            per_allele_hits=dict(hla.per_allele_hits),
            deimmunization_targets=deimmunization_targets,
            hla_personalized=hla_personalized,
            exclusion_reasons=exclusions,
            notes=notes,
        )

    # -- step 2: pick the payload ----------------------------------------
    def select_transgene(
        self, panel: PatientBloodPanel, indication: IschemicIndication
    ) -> TransgeneSpec:
        flags = set(panel.genotype_flags)

        # Monogenic FH is the one clearly actionable genotype in this space.
        if flags & {"LDLR_pathogenic"}:
            spec = self._spec(Transgene.LDLR)
            if indication in spec.indications or panel.ldl_mg_dl >= 190.0:
                return spec
        if flags & {"PCSK9_gof", "APOB_pathogenic"} or panel.ldl_mg_dl >= 190.0:
            return self._spec(Transgene.PCSK9_KD)

        for spec in TRANSGENE_CATALOG:
            if indication in spec.indications:
                return spec

        # Fall back to the angiogenesis payload, the broadest indication set.
        return self._spec(Transgene.VEGF_A165)

    # -- step 3: assemble the regimen ------------------------------------
    def personalize(
        self,
        panel: PatientBloodPanel,
        indication: IschemicIndication = IschemicIndication.CHRONIC_ISCHEMIC_CARDIOMYOPATHY,
    ) -> PersonalizedRegimen:
        # Payload must be chosen *before* the epitope screen, otherwise the
        # immunogenicity burden reflects unrelated transgenes.
        spec = self.select_transgene(panel, indication)
        assessment = self.assess(panel, spec)

        decoy = self._stoich.optimize_ratio(
            panel.nab_titer, dose_vg_per_kg=self.dose_vg_per_kg, weight_kg=panel.weight_kg
        )

        # Modality: switch to redosable mRNA when the AAV immune barrier is high.
        modality = spec.modality_preference
        if assessment.nab_titer >= NAB_EXCLUSION_TITER and modality is Modality.AAV9:
            modality = Modality.LNP_MRNA

        # Tier assignment.
        if not assessment.eligible:
            tier = PersonalizationTier.NOT_ELIGIBLE
        elif assessment.immunogenicity_burden == "high" and assessment.hla_personalized:
            # Bespoke epitope editing is only justified when the screen was run
            # against this patient's actual haplotype. A cohort-level supertype
            # screen says nothing patient-specific, so it must not escalate the
            # tier - otherwise every patient would be priced as bespoke.
            tier = PersonalizationTier.HLA_PERSONALIZED
        elif spec.payload_kb > 4.4:
            tier = PersonalizationTier.BESPOKE
        else:
            tier = PersonalizationTier.OFF_THE_SHELF

        edits: List[str] = []
        if tier is PersonalizationTier.HLA_PERSONALIZED:
            # Prefer the patient-specific targets; fall back to cohort-level ones.
            edits = assessment.deimmunization_targets or self._epitope_edit_targets(spec)

        # Capsid + complement mitigation.
        if assessment.complement_risk >= self.complement_risk_threshold:
            capsid = "charge-masked VR-IV + N87 glycan shield"
            complement_mitigation = "prophylactic complement inhibition (C5) + staggered infusion"
            immunosuppression = "short-course corticosteroid taper"
        else:
            capsid = "engineered AAV9 (epitope charge-masked)"
            complement_mitigation = "none predicted"
            immunosuppression = "none predicted"

        # Promoter: cardiac-restricted, never ubiquitous, to avoid hepatic toxicity
        # (the failure mode that put a real AAV9 cardiac trial on hold).
        promoter = "MHC + dual enhancer + SMAR insulator, miR-122/142/1/208 detarget"

        probe: NAbAssayWell = self._nab.simulate_well(
            donor_id=panel.patient_id,
            serum_titer=panel.nab_titer,
            capsid_type="charge_masked",
            decoy_ratio=decoy.optimal_empty_full_ratio,
        )

        rationale = [
            f"Titer {panel.nab_titer:.0f} classified '{assessment.titer_classification}' "
            f"-> empty:full decoy ratio {decoy.optimal_empty_full_ratio:.0f}:1",
            f"Payload epitope screen: {assessment.hla_epitope_hits} high-affinity hit(s) "
            f"across {len(spec.epitope_segments)} windows; strongest IC50 "
            f"{assessment.strongest_binder_ic50_nm:.0f} nM",
            f"Indication {indication.value} -> transgene {spec.transgene.value} "
            f"({spec.rationale})",
            f"Immunogenicity burden '{assessment.immunogenicity_burden}' "
            f"({assessment.hla_epitope_hits}/{len(spec.epitope_segments)} screened windows "
            f"present, strongest IC50 {assessment.strongest_binder_ic50_nm:.0f} nM)",
            f"Modality {modality.value} selected on immune-barrier grounds",
        ]

        warnings = [
            "All values are computational predictions. No wet-lab or clinical validation.",
            "Not a cure. Not clinically actionable. Danon is a modeling platform, not a therapy.",
        ]
        if panel.hla_alleles and not assessment.hla_personalized:
            warnings.append(
                "Haplotype was only partly resolvable against the modelled supertype "
                "library, so epitope editing is not fully patient-specific."
            )
        if panel.hla_alleles:
            warnings.append(
                "Anchor tables are supertype approximations, not a trained predictor "
                "(NetMHCpan/MHCflurry); treat per-allele calls as indicative."
            )
        if tier is PersonalizationTier.BESPOKE:
            warnings.append(
                "Bespoke tier implies multi-week manufacturing turnaround and "
                "per-patient release testing; cost multiplier excludes release assays."
            )

        return PersonalizedRegimen(
            patient_id=panel.patient_id,
            tier=tier,
            modality=modality,
            transgene=spec.transgene,
            promoter_strategy=promoter,
            dose_vg_per_kg=self.dose_vg_per_kg,
            empty_full_ratio=float(decoy.optimal_empty_full_ratio),
            ratio_range=tuple(decoy.recommended_ratio_range),
            capsid_strategy=capsid,
            epitope_edit_required=tier is PersonalizationTier.HLA_PERSONALIZED,
            epitope_edit_targets=edits,
            complement_mitigation=complement_mitigation,
            immune_suppression=immunosuppression,
            predicted_free_nab_fraction=float(decoy.predicted_free_nab_fraction),
            predicted_complement_risk=float(decoy.complement_activation_risk),
            manufacturing_cost_multiplier=float(decoy.manufacturing_cost_ratio),
            rationale=rationale,
            warnings=warnings,
        )

    # -- helpers ----------------------------------------------------------
    def _spec(self, transgene: Transgene) -> TransgeneSpec:
        for spec in TRANSGENE_CATALOG:
            if spec.transgene is transgene:
                return spec
        raise KeyError(f"transgene {transgene!r} not in catalog")

    @staticmethod
    def _default_segments() -> Sequence[str]:
        """Segment set used for the cohort-level epitope screen."""
        segs: List[str] = []
        for spec in TRANSGENE_CATALOG:
            segs.extend(spec.epitope_segments)
        return segs

    def _epitope_edit_targets(self, spec: TransgeneSpec) -> List[str]:
        """Return the specific windows that must be de-immunized."""
        result = self._hla.evaluate(list(spec.epitope_segments))
        return [b.core_9mer for b in result.binders if b.immunogenic]

    def cohort_summary(
        self, panels: Sequence[PatientBloodPanel], indication: IschemicIndication
    ) -> Dict[str, object]:
        """Tier distribution across a patient panel (for trial design / posters)."""
        tiers: Dict[str, int] = {}
        regimens: List[PersonalizedRegimen] = []
        for p in panels:
            r = self.personalize(p, indication)
            tiers[r.tier.value] = tiers.get(r.tier.value, 0) + 1
            regimens.append(r)
        ratios = [r.empty_full_ratio for r in regimens]
        return {
            "n_patients": len(panels),
            "tier_distribution": tiers,
            "mean_empty_full_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0,
            "bespoke_fraction": (
                tiers.get(PersonalizationTier.BESPOKE.value, 0) / len(panels)
                if panels
                else 0.0
            ),
            "ineligible_fraction": (
                tiers.get(PersonalizationTier.NOT_ELIGIBLE.value, 0) / len(panels)
                if panels
                else 0.0
            ),
        }


def print_personalization_disclaimer() -> None:
    print(
        "=" * 70 + "\n"
        "COMPUTATIONAL MODEL ONLY — NOT MEDICAL ADVICE\n"
        "This personalization layer produces simulation outputs. It has not been\n"
        "validated in vitro or in vivo. No output may be used to treat a patient.\n"
        "Ischemic heart disease is polygenic; personalization here is limited to\n"
        "immune compatibility and dose, not correction of a causal gene defect.\n"
        + "=" * 70
    )
