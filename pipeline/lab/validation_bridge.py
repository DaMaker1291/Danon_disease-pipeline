import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationCandidate:
    candidate_id: int
    candidate_type: str  # "aav" or "lnp"
    sequence: str = ""
    composition: dict = field(default_factory=dict)
    ai_score: float = 0.0
    ai_predictions: dict = field(default_factory=dict)


@dataclass
class CROProtocol:
    protocol_id: str
    created_at: str
    candidates: list
    synthesis_method: str
    target_tissues: list
    organoid_type: str
    readouts: list
    estimated_cost_usd: float
    estimated_timeline_weeks: int


class CROProtocolGenerator:
    def __init__(self):
        self.cros = {
            "synthego": {
                "name": "Synthego",
                "specialties": ["AAV production", "CRISPR"],
                "min_batch": 1e12,
                "cost_per_vg": 0.001,
            },
            "twist_bioscience": {
                "name": "Twist Bioscience",
                "specialties": ["gene synthesis", "AAV packaging"],
                "min_batch": 1e9,
                "cost_per_vg": 0.005,
            },
            "charles_river": {
                "name": "Charles River Laboratories",
                "specialties": ["in vivo studies", "organoid testing"],
                "min_organoids": 100,
                "cost_per_organoid": 500,
            },
            "catalent": {
                "name": "Catalent",
                "specialties": ["LNP formulation", "GMP manufacturing"],
                "min_batch_ml": 100,
                "cost_per_ml": 25,
            },
        }

    def generate_synthesis_protocol(self, candidates: list,
                                     target_tissues: list = None) -> CROProtocol:
        target_tissues = target_tissues or ["cardiac", "neuronal", "joint_cartilage"]

        aav_candidates = [c for c in candidates if c.candidate_type == "aav"]
        lnp_candidates = [c for c in candidates if c.candidate_type == "lnp"]

        estimated_cost = 0.0
        if aav_candidates:
            estimated_cost += len(aav_candidates) * 5000
        if lnp_candidates:
            estimated_cost += len(lnp_candidates) * 2000

        protocol = CROProtocol(
            protocol_id=f"LONGEVITY-{datetime.now().strftime('%Y%m%d')}-{len(candidates)}",
            created_at=datetime.now().isoformat(),
            candidates=[{
                "id": c.candidate_id,
                "type": c.candidate_type,
                "ai_score": c.ai_score,
                "sequence": c.sequence[:100] if c.sequence else "",
                "composition": c.composition,
            } for c in candidates],
            synthesis_method="AAV: HEK293T transfection | LNP: Microfluidic mixing",
            target_tissues=target_tissues,
            organoid_type="iPSC-derived aged tissue organoids",
            readouts=[
                "transduction_efficiency",
                "cell_viability",
                "gene_expression_RNAseq",
                "protein_localization_immunofluorescence",
                "senescence_markers_SABG",
                "dna_damage_response_gamma_H2AX",
            ],
            estimated_cost_usd=estimated_cost,
            estimated_timeline_weeks=8,
        )

        return protocol

    def generate_organoid_testing_plan(self, protocol: CROProtocol) -> dict:
        plan = {
            "protocol_id": protocol.protocol_id,
            "phase_1_vitro": {
                "name": "In Vitro Organoid Screening",
                "duration_weeks": 4,
                "organoids_per_candidate": 24,
                "replicates": 3,
                "conditions": ["24h_expression", "48h_expression", "72h_expression"],
                "readouts": [
                    {"name": "transduction_efficiency", "method": "flow_cytometry"},
                    {"name": "cell_viability", "method": "live_dead_staining"},
                    {"name": "senescence_markers", "method": "SA_beta_galactosidase"},
                    {"name": "dna_damage", "method": "gamma_H2AX_foci"},
                    {"name": "gene_expression", "method": "bulk_RNAseq"},
                ],
                "pass_criteria": {
                    "transduction_efficiency": ">30%",
                    "cell_viability": ">80%",
                    "senescence_reduction": ">20%",
                    "dna_damage": "<baseline",
                },
            },
            "phase_2_targeting": {
                "name": "Tissue Targeting Validation",
                "duration_weeks": 3,
                "tissues": protocol.target_tissues,
                "method": "fluorescent_barcoded_LNP_injection",
                "readouts": [
                    "tissue_accumulation_biodistribution",
                    "organ_specific_transduction",
                    "off_target_effects",
                ],
            },
            "phase_3_safety": {
                "name": "Safety Assessment",
                "duration_weeks": 2,
                "tests": [
                    "cytotoxicity_MTT_assay",
                    "hemolysis_test",
                    "complement_activation",
                    "cytokine_panel_IL6_IL1beta_TNFalpha",
                    "micronucleus_test_genotoxicity",
                ],
            },
        }
        return plan

    def generate_cost_estimate(self, num_candidates: int, num_tissues: int) -> dict:
        aav_cost_per_candidate = 5000
        lnp_cost_per_candidate = 2000
        organoid_cost_per_tissue = 2000
        sequencing_cost_per_sample = 300

        total_candidates = num_candidates
        total_organoids = total_candidates * num_tissues * 24 * 3
        total_sequencing = total_candidates * num_tissues * 3

        estimate = {
            "aav_synthesis": total_candidates * aav_cost_per_candidate,
            "lnp_synthesis": total_candidates * lnp_cost_per_candidate,
            "organoid_culture": total_organoids * 15,
            "sequencing": total_sequencing * sequencing_cost_per_sample,
            "reagents_and_consumables": total_candidates * 500,
            "labor": 15000,
            "total_estimated_usd": 0,
            "breakdown": {
                "synthesis": "AAV: HEK293T transfection + purification | LNP: Microfluidic mixing",
                "testing": "iPSC-derived aged tissue organoids (3 timepoints)",
                "sequencing": "Bulk RNA-seq + flow cytometry",
            },
        }
        estimate["total_estimated_usd"] = sum(
            v for k, v in estimate.items() if isinstance(v, (int, float))
        )

        return estimate

    def export_protocol_package(self, protocol: CROProtocol,
                                 plan: dict, estimate: dict,
                                 output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)

        protocol_path = os.path.join(output_dir, "cro_protocol.json")
        with open(protocol_path, "w") as f:
            json.dump({
                "protocol_id": protocol.protocol_id,
                "created_at": protocol.created_at,
                "synthesis_method": protocol.synthesis_method,
                "target_tissues": protocol.target_tissues,
                "organoid_type": protocol.organoid_type,
                "readouts": protocol.readouts,
                "estimated_cost_usd": protocol.estimated_cost_usd,
                "estimated_timeline_weeks": protocol.estimated_timeline_weeks,
                "candidates": protocol.candidates,
            }, f, indent=2)

        plan_path = os.path.join(output_dir, "testing_plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)

        estimate_path = os.path.join(output_dir, "cost_estimate.json")
        with open(estimate_path, "w") as f:
            json.dump(estimate, f, indent=2)

        summary_path = os.path.join(output_dir, "VALIDATION_SUMMARY.md")
        with open(summary_path, "w") as f:
            f.write(f"# Wet-Lab Validation Package\n\n")
            f.write(f"**Protocol ID:** {protocol.protocol_id}\n")
            f.write(f"**Created:** {protocol.created_at}\n")
            f.write(f"**Candidates:** {len(protocol.candidates)}\n")
            f.write(f"**Target Tissues:** {', '.join(protocol.target_tissues)}\n")
            f.write(f"**Estimated Cost:** ${estimate['total_estimated_usd']:,.0f}\n")
            f.write(f"**Timeline:** {protocol.estimated_timeline_weeks} weeks\n\n")
            f.write(f"## Synthesis Method\n{protocol.synthesis_method}\n\n")
            f.write(f"## Testing Plan\n")
            f.write(f"Phase 1: {plan['phase_1_vitro']['name']} ({plan['phase_1_vitro']['duration_weeks']} weeks)\n")
            f.write(f"Phase 2: {plan['phase_2_targeting']['name']} ({plan['phase_2_targeting']['duration_weeks']} weeks)\n")
            f.write(f"Phase 3: {plan['phase_3_safety']['name']} ({plan['phase_3_safety']['duration_weeks']} weeks)\n\n")
            f.write(f"## Pass Criteria\n")
            for criterion, threshold in plan["phase_1_vitro"]["pass_criteria"].items():
                f.write(f"- {criterion}: {threshold}\n")

        logger.info("Validation package exported to %s", output_dir)
        return output_dir
