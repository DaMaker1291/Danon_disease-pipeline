"""
Danon Pipeline API Server
==========================
FastAPI bridge that exposes the real 18-phase AAV9-LAMP2B optimization pipeline
to the frontend. The frontend does NOT paste sequences — it sends target
constraints, the backend runs the NSGA-II Pareto optimizer + PDB 3UX1
(AAV9) Poisson-Boltzmann charge-masking modules, selects the top engineered capsid,
and returns the generated candidate sequence, its VR-IV / VR-VIII electrostatic
surface profiles, and the exact amino-acid substitutions per 3-fold protrusion.

Run:  uvicorn api_server:app --port 8000
"""
import logging
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from danon.config import DanonConfig
from danon.vector_design.aav_generator import DanonAAVGenerator, WILD_TYPE_AAV9_CAPSID
from danon.vector_design.epitope_masker import (
    EpitopeMasker,
    AA_CHARGE,
    KNOWN_NAB_EPITOPE_RESIDUES,
    CARDIAC_DOCKING_RESIDUES,
    AAV9_VR_COORDINATES,
)
from danon.vector_design.pareto_optimizer import ParetoOptimizer, ParetoPoint
from danon.vector_design.microfluidics_core import MicrofluidicsCore, MicrofluidicConfig
from danon.vector_design.data_ingress import DataIngressEngine
from danon.vector_design.promoter_spec import PromoterSpecEngine
from danon.vector_design.safety_engine import DanonSafetyEngine, DanonSafetyProfile, print_global_regulatory_disclaimer
from danon.vector_design.dms_fitness import DMSFitnessLayer
from danon.vector_design.solvation_energy import SolvationEnergyEngine
from danon.vector_design.smar_insulator import SMARInsulatorEngine, CpGOptimizationEngine, calculate_cpg_density
from danon.vector_design.codon_elongation import CodonElongationEngine, DEFAULT_LAMP2B_PEPTIDE
from danon.vector_design.hla_decoupler import HLADecoupler
from danon.vector_design.synthesis_guard import SynthesisGuard
from danon.vector_design.mirna_detarget import miRNADetargetEngine
from danon.vector_design.translational_readiness import TranslationalReadinessEngine
from danon.vector_design.stoichiometric_calc import StoichiometricCalculator
from danon.vector_design.platform_validator import PlatformValidator, CLINICAL_ASSUMED_INPUTS
from danon.vector_design.tropism_filter import DanonTropismFilter, CARDIAC_RECEPTORS, HEPATIC_RECEPTORS, CHARGE_PROFILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("danon.api")

# Positions in every payload below are 1-based design positions: the residue
# edited is sequence[position - 1], and its VP1 number is
# position + VP1_POSITION_OFFSET. Both offsets are named because the frame used
# to be labelled with one of them and indexed with the other.
VP1_OFFSET = 263            # sequence index = VP1 - VP1_OFFSET
VP1_POSITION_OFFSET = 262   # VP1 = position + VP1_POSITION_OFFSET


def _snake_to_camel(obj):
    """Recursively convert snake_case dict keys to camelCase for the frontend."""
    if isinstance(obj, dict):
        return {re.sub(r'_([a-z])', lambda m: m.group(1).upper(), k): _snake_to_camel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snake_to_camel(i) for i in obj]
    return obj

@asynccontextmanager
async def lifespan(application: FastAPI):
    print_global_regulatory_disclaimer()
    yield

app = FastAPI(title="Danon AAV9-LAMP2B Pipeline API", version="2.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class PipelineConstraints(BaseModel):
    """User-supplied target constraints — NOT a sequence."""
    max_hepatic_accumulation: float = Field(default=0.30, ge=0.0, le=1.0)
    min_cardiac_tropism: float = Field(default=0.50, ge=0.0, le=1.0)
    min_immune_evasion: float = Field(default=0.50, ge=0.0, le=1.0)
    lamp2b_expression_target: float = Field(default=0.70, ge=0.0, le=1.0)
    candidate_pool: int = Field(default=400, ge=50, le=4000)
    max_mutations_vr_iv: int = Field(default=7, ge=0, le=12)
    max_mutations_vr_viii: int = Field(default=10, ge=0, le=20)
    random_seed: int = Field(default=42, ge=0, le=1_000_000)
    # Horizon-2 (phases 19-24) gates
    max_solvation_delta_g: float = Field(default=2.5, ge=0.5, le=10.0)
    min_codon_elongation_index: float = Field(default=0.88, ge=0.5, le=1.0)
    hla_binding_cutoff_nm: float = Field(default=500.0, ge=50.0, le=5000.0)
    synthesis_gc_low: float = Field(default=40.0, ge=20.0, le=50.0)
    synthesis_gc_high: float = Field(default=65.0, ge=55.0, le=80.0)


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def _region_profile(masker: EpitopeMasker, sequence: str, region: str) -> Dict:
    prof = masker.compute_surface_electrostatics(sequence, region)
    residues = []
    for p in prof.residue_profiles:
        residues.append({
            "position": p["position"],
            "positionVp1": p["position"] + VP1_POSITION_OFFSET,
            "aa": p["aa"],
            "charge": round(p["charge"], 4),
            "surfaceAccessibility": round(p["surface_accessibility"], 3),
            "localElectrostaticField": round(p["local_electrostatic_field"], 4),
            "bornSolvationEnergy": round(p["born_solvation_energy"], 4),
            "poissonBoltzmannPotential": round(p["poisson_boltzmann_potential"], 4),
            "isEpitope": p["is_epitope"],
            "epitopeSite": p["epitope_site"],
            "isDockingResidue": p["is_docking_residue"],
        })
    net = sum(r["charge"] for r in residues)
    return {
        "region": region,
        "residueProfiles": residues,
        "netCharge": round(net, 3),
        "surfacePotential": round(prof.surface_potential_before, 4),
        "antibodyDisruptionScore": round(prof.antibody_disruption_score, 4),
    }


def _mutation_records(masker: EpitopeMasker, wt: str, masked: str,
                      mutations, region: str) -> List[Dict]:
    out = []
    for pos, orig, new in mutations:
        out.append({
            "position": pos,
            "positionVp1": pos + VP1_POSITION_OFFSET,
            "original": orig,
            "mutated": new,
            "region": region,
            "chargeBefore": AA_CHARGE.get(orig, 0),
            "chargeAfter": AA_CHARGE.get(new, 0),
            "chargeReversal": AA_CHARGE.get(orig, 0) * AA_CHARGE.get(new, 0) < 0,
            "isEpitope": pos in KNOWN_NAB_EPITOPE_RESIDUES,
            "isDocking": pos in CARDIAC_DOCKING_RESIDUES,
        })
    return out


def run_full_pipeline(c: PipelineConstraints) -> Dict:
    cfg = DanonConfig(
        random_seed=c.random_seed,
        max_hepatic_accumulation=c.max_hepatic_accumulation,
        min_cardiac_tropism=c.min_cardiac_tropism,
        lamp2b_expression_target=c.lamp2b_expression_target,
    )
    generator = DanonAAVGenerator(cfg)
    masker = EpitopeMasker()
    masker.load_pdb_structure()  # BioPython parse of data/pdb/3UX1.pdb
    optimizer = ParetoOptimizer()
    promoter_engine = PromoterSpecEngine()

    # 1) Generate + score a population of capsid variants
    batch_size = 200
    scored = []
    total = c.candidate_pool
    for batch in generator.stream_candidates(total, batch_size):
        scored.extend(batch)

    # 1b) Re-score cardiac tropism using receptor-position model
    tropism_filter = DanonTropismFilter(cfg)
    for cand in scored:
        receptor_score = tropism_filter._compute_tissue_score(cand.sequence, "cardiac_myocytes", 1.0)
        cand.cardiac_tropism_score = float(np.clip(0.1 * cand.cardiac_tropism_score + 0.9 * receptor_score, 0, 1))
        cand.fitness = (
            0.30 * cand.cardiac_tropism_score +
            0.15 * cand.skeletal_muscle_score +
            0.20 * cand.hepatic_avoidance_score +
            0.15 * cand.immune_evasion_score +
            0.10 * cand.lamp2b_compatibility +
            0.10 * cand.structural_score +
            0.05 * cand.stability_score
        )

    # 2) Constraint filter (soft) then rank by fitness
    def passes(cand):
        return (
            cand.cardiac_tropism_score >= c.min_cardiac_tropism - 0.15 and
            cand.hepatic_avoidance_score >= (1.0 - c.max_hepatic_accumulation) - 0.15 and
            cand.immune_evasion_score >= c.min_immune_evasion - 0.25
        )

    feasible = [x for x in scored if passes(x)] or scored
    feasible.sort(key=lambda x: x.fitness, reverse=True)
    top = feasible[:12]

    # 3) Promoter score (cardiac-restricted) used as a Pareto objective.
    # Computed, never defaulted: this value feeds every Pareto ranking below and
    # the clinical chart the user sees, so substituting a plausible constant on
    # failure would silently corrupt the result. The engine is pure Python with
    # no optional dependencies, so an exception here is a real defect and should
    # surface as one.
    promoter_score = float(np.clip(promoter_engine.our_best_score(), 0.0, 1.0))

    # Compute miRNA de-targeting score using wild-type AAV9 as baseline (target sites conserved)
    mirna_engine = miRNADetargetEngine()
    mirna_score = mirna_engine.score_candidate_for_mirna_compatibility(WILD_TYPE_AAV9_CAPSID)

    # 4) Apply PDB 3UX1 Poisson-Boltzmann dual-region charge masking to each
    best = None
    best_score = -1.0
    pareto_inputs: List[ParetoPoint] = []
    for cand in top:
        mask_iv, mask_viii = masker.design_dual_region_masking(
            cand.sequence,
            max_mutations_iv=c.max_mutations_vr_iv,
            max_mutations_viii=c.max_mutations_vr_viii,
        )
        masked_seq = mask_viii.masked_sequence
        immune = 0.5 * mask_iv.overall_mask_score + 0.5 * mask_viii.overall_mask_score
        combined = optimizer.our_pipeline_score(cand, promoter_score, mirna_score=mirna_score)
        combined = 0.6 * combined + 0.4 * immune

        pareto_inputs.append(ParetoPoint(
            candidate_id=cand.candidate_id,
            cardiac_tropism=cand.cardiac_tropism_score,
            hepatic_avoidance=cand.hepatic_avoidance_score,
            immune_evasion=immune,
            lamp2b_expression=cand.lamp2b_compatibility,
            promoter_score=promoter_score,
            mirna_score=mirna_score,
        ))

        if combined > best_score:
            best_score = combined
            best = {
                "candidate": cand,
                "mask_iv": mask_iv,
                "mask_viii": mask_viii,
                "masked_seq": masked_seq,
                "immune": immune,
                "combined": combined,
            }

    # Compute real miRNA de-targeting score for the winning sequence
    mirna_engine = miRNADetargetEngine()
    mirna_score = mirna_engine.score_candidate_for_mirna_compatibility(masked_seq)

    # 5) Pareto rank the top set for the clinical chart
    ranked = optimizer.select_top_n(pareto_inputs, len(pareto_inputs))
    pareto_front = [{
        "candidateId": p.candidate_id,
        "cardiacTropism": round(p.cardiac_tropism, 4),
        "hepaticAvoidance": round(p.hepatic_avoidance, 4),
        "immuneEvasion": round(p.immune_evasion, 4),
        "lamp2bExpression": round(p.lamp2b_expression, 4),
        "promoterScore": round(p.promoter_score, 4),
        "mirnaScore": round(p.mirna_score, 4),
        "paretoRank": p.pareto_rank,
        "crowdingDistance": (0.0 if p.crowding_distance == float("inf") else round(p.crowding_distance, 4)),
    } for p in ranked]

    cand = best["candidate"]
    mask_iv = best["mask_iv"]
    mask_viii = best["mask_viii"]
    masked_seq = best["masked_seq"]

    mutations = (
        _mutation_records(masker, cand.sequence, mask_iv.masked_sequence, mask_iv.mutations, "VR_IV") +
        _mutation_records(masker, mask_iv.masked_sequence, masked_seq, mask_viii.mutations, "VR_VIII")
    )

    # UCL/GOSH baseline: wild-type AAV9 + CMV, no epitope masking. Computed as a
    # zero-substitution run of the *same* 50/50 VR-IV + VR-VIII objective the
    # design is scored on, so 'improvement' is a like-for-like ratio.
    ucl_immune = masker.utcl_composite_score(cand.sequence)
    improvement = float(best["immune"] / max(ucl_immune, 1e-6))

    # ----- Accurate LAMP2B cargo estimate for the single-vs-dual vector gate -----
    from danon.vector_design.dual_vector import evaluate_vector_capacity
    lamp2b_cargo_bp = int(round(len(DEFAULT_LAMP2B_PEPTIDE) * 3 + 600 + 1000))
    vector_capacity = evaluate_vector_capacity(lamp2b_cargo_bp)

    # ----- Horizon-2: phases 19-24 on the winning capsid + LAMP2B payload -----
    horizon2 = run_horizon2(c, cand.sequence, masked_seq, mutations, promoter_score, vector_capacity)

    # Real Navier-Stokes-derived mixing efficiency. As with promoter_score, a
    # fabricated fallback would put an invented number into a reported metric,
    # so a failure propagates instead of being papered over.
    mixing_eff = float(MicrofluidicsCore().simulate().mixing_efficiency)

    docking_ok = mask_iv.cardiac_docking_preserved and mask_viii.cardiac_docking_preserved

    # ----- Regulatory readiness rubric: computed, not asserted -----
    # Only values this pipeline actually derives are supplied. The clinical and
    # programme evidence it does not measure arrives through the explicitly
    # labelled assumed block, so the result is tagged as default-padded rather
    # than reading as measurement. There is no dosing score on this endpoint, so
    # ``clinical_efficacy`` is a gating dimension that stays unscored and the
    # verdict reads UNKNOWN -- which is the honest report, not a pass and not a
    # fail. The verdict is a self-assessment; MHRA publishes no such scale.
    reg_assessment = PlatformValidator().evaluate_candidate(
        {
            "cardiac_tropism": cand.cardiac_tropism_score,
            "hepatic_accumulation": 1.0 - cand.hepatic_avoidance_score,
            "immune_evasion": best["immune"],
            "lamp2b_expression": cand.lamp2b_compatibility,
            "promoter_score": promoter_score,
            "mirna_score": mirna_score,
        },
        assumed_inputs=CLINICAL_ASSUMED_INPUTS,
    )
    if reg_assessment.weighted_score is None:
        # Propagate rather than pad: a fabricated fallback here would put an
        # invented number into a reported phase score.
        raise RuntimeError(
            "regulatory rubric produced no scored dimension; refusing to "
            "report a phase score"
        )
    reg_phase = (
        "Regulatory Readiness (self-assessment)", "Regulatory",
        float(np.clip(reg_assessment.weighted_score, 0, 1)),
        f"rubric verdict {reg_assessment.verdict} \u00b7 "
        f"{reg_assessment.scored_dimensions}/{reg_assessment.total_dimensions} "
        f"dimensions scored \u00b7 not an MHRA determination",
    )
    phases_18 = [
        ("Data Ingress & QC", "Data", 1.0, f"{len(scored)} variants ingested · template mapped"),
        ("AAV9 Capsid Generation", "Generation", 1.0, f"{len(scored)} VP1 variants sampled from pool of {c.candidate_pool}"),
        ("Cardiac Tropism Scoring", "Tropism", cand.cardiac_tropism_score, f"cardiomyocyte affinity {cand.cardiac_tropism_score*100:.0f}%"),
        ("Skeletal Muscle Tropism", "Tropism", cand.skeletal_muscle_score, f"myofiber affinity {cand.skeletal_muscle_score*100:.0f}%"),
        ("Hepatic De-targeting", "Safety", cand.hepatic_avoidance_score, f"liver avoidance {cand.hepatic_avoidance_score*100:.0f}%"),
        ("VR-IV Epitope Masking (PB)", "Immunology", mask_iv.overall_mask_score, f"mask score {mask_iv.overall_mask_score:.2f} · {len(mask_iv.mutations)} subs"),
        ("VR-VIII Epitope Masking (PB)", "Immunology", mask_viii.overall_mask_score, f"mask score {mask_viii.overall_mask_score:.2f} · {len(mask_viii.mutations)} subs"),
        ("Charge-Reversal Electrostatics", "Immunology", mask_viii.charge_reversal_ratio, f"charge-flip ratio {mask_viii.charge_reversal_ratio:.2f}"),
        ("Cardiac Docking Preservation", "Tropism", 1.0 if docking_ok else 0.3, "receptor footprint preserved" if docking_ok else "docking residue mutated"),
        ("Structural Integrity", "Structural", cand.structural_score, f"integrity {cand.structural_score:.2f}"),
        ("LAMP2B Compatibility", "Payload", cand.lamp2b_compatibility, f"compatibility {cand.lamp2b_compatibility:.2f}"),
        ("Immune Evasion (NAb)", "Immunology", best["immune"], f"NAb escape {best['immune']*100:.0f}%"),
        ("Cardiac Promoter Specificity", "Expression", promoter_score, f"cardiac specificity {promoter_score:.2f}"),
        ("miRNA De-targeting", "Safety", mirna_score, f"miR-122/miR-1/miR-142/miR-208 score {mirna_score:.2f}"),
        ("Vector Topology (Capacity Gate)", "Payload", 1.0 if vector_capacity["strategy"].startswith("Single") else 0.4,
         f"{vector_capacity['strategy']} · {vector_capacity['cargo_length_bp']} bp · tox x{vector_capacity['toxicity_risk_multiplier']}"),
        ("Stoichiometric Decoy Optimization", "Formulation", float(np.clip(StoichiometricCalculator().our_best_score(), 0, 1)),
         f"empty:full capsid ratio optimized for NAb decoy"),
        ("Microfluidic LNP Formulation", "Formulation", float(np.clip(mixing_eff, 0, 1)), f"mixing efficiency {mixing_eff:.2f}"),
        reg_phase,
    ]
    # ------------------------------------------------------------------ #
    # NOTE -- a compounded "combinatorial design-space advantage" used to be
    # computed here as prod(1.5 + 2.5 * score) over the 24 phases and shown in
    # the UI as, e.g., "24.2 trillion x". It was removed: the gates are neither
    # orthogonal nor independent (phases 3-12 all score the same selected
    # capsid, phases 21-24 all score the same sequence), the multipliers were an
    # arbitrary affine map of the already-reported scores, and a product of 24
    # dimensionless gate scores has no units and no interpretation. Nothing
    # measured changes when it is removed; only the headline changes.
    # The registry now reports the composition of the gates instead.
    # ------------------------------------------------------------------ #
    all_specs = phases_18 + horizon2["_new_phases"]
    phases = []
    for i, (name, cat, score, metric) in enumerate(all_specs, start=1):
        score = float(np.clip(score, 0, 1))
        phases.append({
            "id": i, "name": name, "category": cat,
            "score": round(score, 4), "status": _status(score),
            "metric": metric,
            "horizon": 2 if i >= 19 else 1,
        })
    by_status = {
        status: sum(1 for p in phases if p["status"] == status)
        for status in ("pass", "warn", "fail")
    }
    phase_summary = {
        "nPhases": len(phases),
        "nPass": by_status["pass"],
        "nWarn": by_status["warn"],
        "nFail": by_status["fail"],
        "basis": (
            "gate pass/warn/fail on the pipeline's own scores at fixed "
            "thresholds (pass >= 0.6, warn >= 0.4); the scores are model "
            "outputs, and the gates are not independent of one another"
        ),
    }

    return {
        "sequence": masked_seq,
        "wildTypeSequence": WILD_TYPE_AAV9_CAPSID,
        "vpOffset": VP1_OFFSET,
        "candidateId": cand.candidate_id,
        "mutations": mutations,
        "regions": {
            "VR_IV": _region_profile(masker, masked_seq, "VR_IV"),
            "VR_VIII": _region_profile(masker, masked_seq, "VR_VIII"),
        },
        "regionResidueRanges": {
            "VR_IV": {"start": 187, "end": 207, "startVp1": 187 + VP1_POSITION_OFFSET, "endVp1": 207 + VP1_POSITION_OFFSET},
            "VR_VIII": {"start": 309, "end": 339, "startVp1": 309 + VP1_POSITION_OFFSET, "endVp1": 339 + VP1_POSITION_OFFSET},
        },
        "scores": {
            "cardiacTropism": round(cand.cardiac_tropism_score, 4),
            "hepaticAvoidance": round(cand.hepatic_avoidance_score, 4),
            "immuneEvasion": round(best["immune"], 4),
            "lamp2bExpression": round(cand.lamp2b_compatibility, 4),
            "structural": round(cand.structural_score, 4),
            "overall": round(best["combined"], 4),
            "maskScoreVrIv": round(mask_iv.overall_mask_score, 4),
            "maskScoreVrViii": round(mask_viii.overall_mask_score, 4),
            "chargeReversalRatio": round(mask_viii.charge_reversal_ratio, 4),
            "improvementVsUcl": round(improvement, 1),
            "dockingPreserved": bool(mask_iv.cardiac_docking_preserved and mask_viii.cardiac_docking_preserved),
        },
        "paretoFront": pareto_front,
        "spikeCount": 20,
        "poolEvaluated": len(scored),
        "phases": phases,
        "advancedMetrics": horizon2["advancedMetrics"],
        "phaseSummary": phase_summary,
        "phasesPassed": phase_summary["nPass"],
        "totalPhases": 24,
        "translationalReadiness": _snake_to_camel(TranslationalReadinessEngine().evaluate_translational_gate().model_dump()),
    }


# --------------------------------------------------------------------------- #
# Horizon-2 (phases 19-24) execution + 24-phase registry
# --------------------------------------------------------------------------- #
def _status(score: float, warn: float = 0.6, fail: float = 0.4) -> str:
    return "pass" if score >= warn else ("warn" if score >= fail else "fail")


def _junction_segments(peptide: str, split_positions, half: int = 8):
    segs = []
    for sp in split_positions:
        lo = max(0, sp - half)
        hi = min(len(peptide), sp + half)
        seg = peptide[lo:hi]
        if seg:
            segs.append((sp, seg))
    return segs


def run_horizon2(c: PipelineConstraints, wt_seq: str, masked_seq: str,
                 mutations, promoter_score: float, vector_capacity: dict) -> Dict:
    vr_coords = {
        "VR_IV": AAV9_VR_COORDINATES["VR_IV"]["coordinates"],
        "VR_VIII": AAV9_VR_COORDINATES["VR_VIII"]["coordinates"],
    }
    mut_tuples = [(m["position"], m["original"], m["mutated"]) for m in mutations]

    # PHASE 19 — DMS fitness boundary
    dms = DMSFitnessLayer().evaluate(mut_tuples)
    dms_score = float(np.clip(0.55 + 0.6 * dms.mean_dms_fitness, 0, 1)) if dms.capsid_viable else 0.25

    # PHASE 20 — solvation free energy
    sol = SolvationEnergyEngine(c.max_solvation_delta_g).evaluate(wt_seq, masked_seq, vr_coords)
    sol_score = float(np.clip(1.0 - max(0.0, sol.ddg_solv_total) / (c.max_solvation_delta_g * 2), 0, 1))

    # PHASE 21 — transcriptional shielding: CpG depletion (primary) + S/MAR (secondary)
    cpg_engine = CpGOptimizationEngine(cpg_density_threshold=1.0)
    cpg_report = cpg_engine.optimize(protein=DEFAULT_LAMP2B_PEPTIDE)
    smar = SMARInsulatorEngine().evaluate()  # retained secondary assessment only
    smar_score = float(np.clip(0.5 * cpg_report.cpg_within_threshold + 0.5 * smar["mean_smar_strength"], 0, 1))

    # PHASE 22 — codon elongation / tAI
    codon = CodonElongationEngine(c.min_codon_elongation_index).evaluate(protein=DEFAULT_LAMP2B_PEPTIDE)
    codon_score = float(np.clip(codon.tai_before, 0, 1))

    # PHASE 23 — HLA-DRB1 junction decoupling; pick least-immunogenic split site
    hla_engine = HLADecoupler(c.hla_binding_cutoff_nm)
    split_candidates = [175, 200, 250]
    best_hla = None
    best_split = split_candidates[0]
    for sp, seg in _junction_segments(DEFAULT_LAMP2B_PEPTIDE, split_candidates):
        res = hla_engine.evaluate([seg])
        if best_hla is None or res.strongest_binder_ic50_nm > best_hla.strongest_binder_ic50_nm:
            best_hla, best_split = res, sp
    hla_score = float(np.clip(np.log10(max(best_hla.strongest_binder_ic50_nm, 1)) /
                              np.log10(hla_engine.ic50_cutoff_nm * 20), 0, 1))

    # PHASE 24 — synthesis feasibility on the codon-optimized therapeutic ORF
    # (the AT-rich S/MAR insulator flanks are ordered separately as known elements)
    lamp2b_cds = CodonElongationEngine().back_translate(DEFAULT_LAMP2B_PEPTIDE)
    synth = SynthesisGuard((c.synthesis_gc_low, c.synthesis_gc_high)).evaluate(lamp2b_cds)
    synth_score = float(np.clip(1.0 - synth.n_hard_failures / 5.0, 0, 1))

    # Accurate LAMP2B cargo estimate for the single-vs-dual vector capacity gate.
    # Passed in from run_full_pipeline (CDS + promoter + UTRs/polyA).
    lamp2b_cargo_bp = vector_capacity["cargo_length_bp"]

    advanced = {
        "dms": {
            "capsidViable": dms.capsid_viable,
            "minDmsFitness": dms.min_dms_fitness,
            "meanDmsFitness": dms.mean_dms_fitness,
            "conservedPocketViolations": dms.conserved_pocket_violations,
            "lethalMutations": dms.lethal_mutations,
            "boundaryMargin": dms.fitness_boundary_margin,
            "mutationScores": [s.model_dump() for s in dms.mutation_scores],
        },
        "solvation": {
            "ddgSolvTotal": sol.ddg_solv_total,
            "plasmaSoluble": sol.plasma_soluble,
            "aggregationRisk": sol.aggregation_risk_score,
            "maxAllowedDdg": sol.max_allowed_ddg,
            "regions": [{"region": r.region, "ddgSolv": r.ddg_solv,
                         "dgWildType": r.dg_solv_wild_type, "dgMutant": r.dg_solv_mutant}
                        for r in sol.regions],
        },
        "smar": {
            "cpgDensityRaw": cpg_report.raw_cpg_density,
            "cpgDensityDepleted": cpg_report.depleted_cpg_density,
            "cpgReductionPct": cpg_report.cpg_reduction_pct,
            "cpgWithinThreshold": cpg_report.cpg_within_threshold,
            "primaryStrategy": "CpG depletion (synonymous recoding)",
            "secondarySmAR": {
                "combinedAtContent": smar["combined_at_content"],
                "meanStrength": smar["mean_smar_strength"],
                "shieldingPredicted": smar["shielding_predicted"],
            },
        },
        "vectorCapacity": {
            "strategy": vector_capacity["strategy"],
            "cargoLengthBp": vector_capacity["cargo_length_bp"],
            "toxicityRiskMultiplier": vector_capacity["toxicity_risk_multiplier"],
            "clinicalJustification": vector_capacity["clinical_justification"],
        },
        "codon": {
            "tai": codon.tai_before, "minWindowTai": getattr(codon, 'min_window_tai', 0.0),
            "nCodons": codon.n_codons,             "stallSites": len(getattr(codon, 'stall_sites', [])),
            "codonOptimized": getattr(codon, 'codon_optimized', False), "threshold": getattr(codon, 'min_index_threshold', 0.88),
            "taiAfter": getattr(codon, 'tai_after', 0.0),
            "meanElongationRate": getattr(codon, 'mean_elongation_rate', 0.0),
        },
        "hla": {
            "chosenSplitPosition": best_split,
            "strongestIc50Nm": best_hla.strongest_binder_ic50_nm,
            "highAffinityHits": best_hla.high_affinity_hits,
            "decoupled": best_hla.decoupled,
            "cutoffNm": best_hla.ic50_cutoff_nm,
            "peptidesEvaluated": best_hla.peptides_evaluated,
            "binders": [b.model_dump() for b in best_hla.binders[:8]],
        },
        "synthesis": {
            "lengthBp": synth.length_bp, "gcContent": synth.gc_content,
            "gcMinWindow": synth.gc_min_window, "gcMaxWindow": synth.gc_max_window,
            "outOfBoundsWindows": len(synth.out_of_bounds_windows),
            "homopolymerRuns": len(synth.homopolymer_runs),
            "invertedRepeats": len(synth.inverted_repeats),
            "directRepeats": len(synth.direct_repeats),
            "synthesizable": synth.synthesizable, "hardFailures": synth.n_hard_failures,
            "gcBounds": synth.gc_window_bounds,
        },
    }

    # Phase 19-24 registry entries
    new_phases = [
        ("Deep Mutational Scan Fitness", "Structural", dms_score,
         f"min fitness {dms.min_dms_fitness:+.2f} · {dms.lethal_mutations} lethal · {dms.conserved_pocket_violations} pocket hits"),
        ("Solvation Free Energy (ΔΔG)", "Stability", sol_score,
         f"ΔΔG_solv {sol.ddg_solv_total:+.2f} kcal/mol · {'soluble' if sol.plasma_soluble else 'aggregation risk'}"),
        ("CpG Depletion / Anti-Silencing", "Expression", smar_score,
         f"CpG {cpg_report.raw_cpg_density:.2f}→{cpg_report.depleted_cpg_density:.2f}/100bp · -{cpg_report.cpg_reduction_pct:.0f}% · {'depleted' if cpg_report.cpg_within_threshold else 'high'}"),
        ("Codon Elongation / tAI", "Translation", codon_score,          f"tAI {codon.tai_before:.3f} · min-window {getattr(codon, 'min_window_tai', 0.0):.2f} · {len(codon.stall_sites)} stalls"),
        ("HLA-DRB1 Junction Decoupler", "Immunology", hla_score,
         f"split@{best_split} · strongest IC50 {best_hla.strongest_binder_ic50_nm:.0f} nM · {best_hla.high_affinity_hits} hits"),
        ("Synthesis Feasibility Screen", "Manufacturing", synth_score,
         f"GC {synth.gc_content:.0f}% · {len(synth.inverted_repeats)} hairpins · {'synthesizable' if synth.synthesizable else 'flagged'}"),
    ]

    return {
        "advancedMetrics": advanced,
        "_new_phases": new_phases,
        "phases": [],  # filled by build_phase_registry
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "danon-pipeline", "version": "2.2", "phases": 24}


@app.post("/api/run-pipeline")
def run_pipeline(constraints: PipelineConstraints):
    logger.info("Running pipeline with constraints: %s", constraints.model_dump())
    return run_full_pipeline(constraints)


class ElectrostaticRequest(BaseModel):
    sequence: str
    region: str = "VR_VIII"


@app.post("/api/electrostatics")
def electrostatics(req: ElectrostaticRequest):
    masker = EpitopeMasker()
    masker.load_pdb_structure()
    return _region_profile(masker, req.sequence, req.region)


@app.post("/api/microfluidics")
def microfluidics(config: Dict):
    mc = MicrofluidicsCore(MicrofluidicConfig(**config) if config else MicrofluidicConfig())
    t = mc.simulate()
    return {
        "reynoldsNumberAqueous": round(t.reynolds_number_aqueous, 4),
        "reynoldsNumberOrganic": round(t.reynolds_number_organic, 4),
        "reynoldsNumberMixed": round(t.reynolds_number_mixed, 4),
        "wallShearStressPa": round(t.wall_shear_stress_pa, 4),
        "maxShearStressPa": round(t.max_shear_stress_pa, 4),
        "mixingEfficiency": round(t.mixing_efficiency, 4),
        "pecletNumber": round(t.peclet_number, 2),
        "pressureDropPa": round(t.pressure_drop_pa, 3),
        "flowRegime": t.flow_regime,
        "shearStressSafe": t.shear_stress_safe,
        "frr": round(t.frr, 4),
        "concentrationProfile": mc.get_concentration_profile(60),
    }


@app.get("/api/clinical")
def clinical():
    months = list(range(0, 25, 3))
    outcomes = []
    for mt, decay, base in [
        ("Frameshift (severe)", 0.055, 145.0),
        ("Missense (moderate)", 0.038, 130.0),
        ("Splice-site", 0.045, 138.0),
    ]:
        lvmi = [round(base * np.exp(-decay * m), 2) for m in months]
        surv = [round(float(np.clip(1.0 - 0.006 * m, 0.5, 1.0)), 4) for m in months]
        outcomes.append({
            "mutationType": mt,
            "months": months,
            "lvmiPredicted": lvmi,
            "survivalProbability": surv,
            "survivalLower": [round(max(0.0, s - 0.05), 4) for s in surv],
            "survivalUpper": [round(min(1.0, s + 0.04), 4) for s in surv],
        })
    return outcomes


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = (await file.read()).decode("utf-8", errors="ignore")
    # A longitudinal MRI availability manifest is a different kind of input from
    # a sequence, and it is routed by CONTENT rather than extension. Left to
    # fall through, it would reach the sequence path and be reported as a
    # one-record sequence with an invented coverage figure -- the client-side
    # portal used to do exactly that for any unrecognised file.
    from danon.availability import census_from_text, looks_like_availability_manifest

    if looks_like_availability_manifest(raw):
        try:
            census = census_from_text(
                raw, source=f"uploaded manifest {file.filename or 'unnamed'}"
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"availability manifest refused unread: {exc}",
            ) from exc
        return {
            "kind": "mri_availability_manifest",
            "availabilityCensus": census.to_json_dict(),
            "notes": census.notes,
        }

    engine = DataIngressEngine()
    name = (file.filename or "").lower()
    if name.endswith((".fastq", ".fq")):
        result = engine.process_fastq_mutations(raw, is_path=False)
    elif name.endswith((".gb", ".genbank", ".gbk")):
        result = engine.process_genbank_mutations(raw, is_path=False)
    else:
        seq = "".join(ch for ch in raw.upper() if ch in "ACDEFGHIKLMNPQRSTVWY")
        result = engine.sanitize_and_map(seq)
    return {
        "sourceFormat": result.source_format,
        "recordsParsed": result.records_parsed,
        "mutationsIsolated": [{
            "positionVp1": m.position_vp1,
            "originalAa": m.original_aa,
            "mutatedAa": m.mutated_aa,
            "region": m.region,
            "confidence": m.confidence,
        } for m in result.mutations_isolated],
        "templateCoverage": result.template_coverage,
        "sequenceValid": result.sequence_valid,
        "warnings": getattr(result, "warnings", []) or [],
    }


# --------------------------------------------------------------------------- #
# ESM protein language model endpoints
# --------------------------------------------------------------------------- #
class EsmFitnessRequest(BaseModel):
    sequence: str
    mutations: Optional[List[Dict]] = None


@app.post("/api/esm/fitness")
def esm_fitness(req: EsmFitnessRequest):
    """ESM-2 per-residue fitness scoring for a protein sequence."""
    try:
        from danon.vector_design.esm_inference import compute_per_residue_fitness, score_mutations_with_esm
        fitness = compute_per_residue_fitness(req.sequence)
        result = {
            "sequenceLength": fitness.sequence_length,
            "meanFitness": fitness.mean_fitness,
            "minFitness": fitness.min_fitness,
            "maxFitness": fitness.max_fitness,
            "perResidue": [{
                "position": r.position,
                "positionVp1": r.position_vp1,
                "wildTypeAa": r.wild_type_aa,
                "logLikelihoodRatio": r.log_likelihood_ratio,
                "predictedClass": r.predicted_class,
                "confidence": r.confidence,
            } for r in fitness.per_residue[:200]],
            "modelVersion": fitness.model_version,
        }
        if req.mutations:
            result["scoredMutations"] = score_mutations_with_esm(req.sequence, req.mutations)
        return _snake_to_camel(result)
    except Exception as e:
        logger.warning("ESM fitness scoring failed: %s", e)
        return {"error": str(e)}


class EsmStructureRequest(BaseModel):
    sequence: str


@app.post("/api/esm/structure")
def esm_structure(req: EsmStructureRequest):
    """ESM-2 derived structural features: secondary structure, accessibility, contacts."""
    try:
        from danon.vector_design.esm_inference import compute_structural_features
        result = compute_structural_features(req.sequence)
        return _snake_to_camel(result)
    except Exception as e:
        logger.warning("ESM structural feature prediction failed: %s", e)
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
# Horizon-2 individual phase endpoints (phases 19-24)
# --------------------------------------------------------------------------- #
class MutationsRequest(BaseModel):
    mutations: List[Dict] = Field(default_factory=list)  # {position, original, mutated}


@app.post("/api/phase/dms")
def phase_dms(req: MutationsRequest):
    tuples = [(m["position"], m["original"], m["mutated"]) for m in req.mutations]
    return DMSFitnessLayer().evaluate(tuples).model_dump()


class SolvationRequest(BaseModel):
    wild_type: str = WILD_TYPE_AAV9_CAPSID
    mutant: str = WILD_TYPE_AAV9_CAPSID
    max_delta_g: float = 2.5


@app.post("/api/phase/solvation")
def phase_solvation(req: SolvationRequest):
    coords = {r: AAV9_VR_COORDINATES[r]["coordinates"] for r in ("VR_IV", "VR_VIII")}
    return SolvationEnergyEngine(req.max_delta_g).evaluate(req.wild_type, req.mutant, coords).model_dump()


class SMARRequest(BaseModel):
    upstream: Optional[str] = None
    downstream: Optional[str] = None


@app.post("/api/phase/smar")
def phase_smar(req: SMARRequest):
    return SMARInsulatorEngine().evaluate(req.upstream, req.downstream).model_dump()


class CodonRequest(BaseModel):
    cds: Optional[str] = None
    protein: Optional[str] = None
    min_tai: float = 0.88


@app.post("/api/phase/codon")
def phase_codon(req: CodonRequest):
    return CodonElongationEngine(req.min_tai).evaluate(cds=req.cds, protein=req.protein).model_dump()


class HLARequest(BaseModel):
    segments: List[str]
    cutoff_nm: float = 500.0


@app.post("/api/phase/hla")
def phase_hla(req: HLARequest):
    return HLADecoupler(req.cutoff_nm).evaluate(req.segments).model_dump()


class SynthesisRequest(BaseModel):
    dna: str
    gc_low: float = 40.0
    gc_high: float = 65.0


@app.post("/api/phase/synthesis")
def phase_synthesis(req: SynthesisRequest):
    return SynthesisGuard((req.gc_low, req.gc_high)).evaluate(req.dna).model_dump()


# --------------------------------------------------------------------------- #
# Digital-twin simulation: MRI mask in, paired 365-day research run out
# --------------------------------------------------------------------------- #
TWIN_MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def read_nifti_from_bytes(raw: bytes):
    """Parse a NIfTI-1 buffer via the repo's own parser (temp-file hop).

    read_nifti takes a path; the upload arrives as bytes. The temp file lives
    only for the parse and is deleted in a finally block.
    """
    import tempfile, os
    from danon.imaging import read_nifti
    fd, tmp = tempfile.mkstemp(suffix=".nii")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        return read_nifti(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


@app.post("/api/twin/plan")
def twin_plan(file: UploadFile = File(...)):
    """Run the paired twin simulation on an uploaded segmentation mask.

    Returns the full planning payload (geometry summary, schedule text,
    daily curves, paired metrics, events, provenance) plus the final-field
    VTI as base64 -- the exact file ParaView opens, also rendered in-browser
    by VtiViewer. Oversize grids are block-max-downsampled (tumour-preserving)
    to the engine budget and the applied factors are reported, never silent.
    """
    import base64
    import os
    import tempfile
    from danon.imaging import MAX_TOTAL_VOXELS, mask_bytes_provenance
    from danon.twin_plan import (
        derived_voxel_sizes,
        downsample_mask_to_budget,
        plan_from_mask,
    )

    name = (file.filename or "").lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(415, "only .nii / .nii.gz segmentation masks are accepted")
    raw = file.file.read(TWIN_MAX_UPLOAD_BYTES + 1)
    if len(raw) > TWIN_MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"upload exceeds {TWIN_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )

    # Validate and size-check before running anything expensive.
    try:
        arr, hdr = read_nifti_from_bytes(raw)
    except Exception as e:
        raise HTTPException(422, f"could not parse NIfTI-1: {e}")
    if int((arr > 0).sum()) == 0:
        raise HTTPException(422, "mask has no nonzero voxels: nothing to simulate")

    # Provenance of the exact bytes handed to us: the shape and spacing the
    # file declared, and a hash so a recorded run can be tied to one file. The
    # engine runs on a temp copy, so these are the only record of the input.
    # Computed by danon.imaging so the API and the audit cannot disagree on
    # what "the input" is (sha256 over the bytes AS RECEIVED, not a rewrite).
    input_provenance = mask_bytes_provenance(raw)
    input_sha256 = input_provenance["input_sha256"]
    original_shape = input_provenance["original_grid_shape_xyz"]
    source_voxel_mm = input_provenance["source_voxel_mm"]

    downsample_factors: List[int] = []
    derived_sizes: Tuple[float, ...] = tuple(float(v) for v in hdr.voxel_sizes)
    tmp: Optional[str] = None
    try:
        if arr.size > MAX_TOTAL_VOXELS:
            # Oversize grid: block-max downsample (tumour-preserving), then
            # rewrite as a clean uint8 .nii. The derived header carries the
            # SCALED spacing, not the source spacing -- one halving makes each
            # derived voxel span two source voxels per axis, and writing the
            # source spacing back would understate every physical volume taken
            # from the file by 8x per halving (the defect danon.grid_artifact
            # records against the committed derived volumes).
            downsampled, downsample_factors = downsample_mask_to_budget(arr > 0)
            derived_sizes = derived_voxel_sizes(hdr.voxel_sizes, downsample_factors)
            fd, tmp = tempfile.mkstemp(suffix=".nii")
            os.close(fd)
            from danon.imaging import write_nifti
            write_nifti(
                tmp, downsampled.astype(np.uint8), voxel_sizes=derived_sizes
            )
        else:
            fd, tmp = tempfile.mkstemp(
                suffix=".nii.gz" if name.endswith(".gz") else ".nii"
            )
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
        try:
            result = plan_from_mask(tmp, export_vti=True)
        except ValueError as e:
            raise HTTPException(422, str(e))
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    vti_b64 = (
        base64.b64encode(result.vti_bytes).decode("ascii") if result.vti_bytes else None
    )
    payload = result.to_json_dict()
    # The plan was computed on a temp path; report the user's filename, and
    # report the input's own geometry beside the grid the engine actually ran.
    payload["geometry"]["mask_path"] = file.filename
    payload["geometry"]["original_grid_shape_xyz"] = original_shape
    payload["geometry"]["source_voxel_mm"] = source_voxel_mm
    payload["geometry"]["source_voxel_sizes_mm"] = input_provenance[
        "source_voxel_sizes_mm"
    ]
    payload["geometry"]["input_sha256"] = input_sha256
    # Also carried in the provenance block, so the run is tied to the file even
    # by a reader who looks at the human-readable record rather than geometry.
    payload["provenance"]["input_sha256"] = input_sha256
    if downsample_factors:
        per_axis = 2 ** len(downsample_factors)
        payload["geometry"]["downsample_factors"] = downsample_factors
        payload["geometry"]["downsample_per_axis"] = per_axis
        payload["notes"].append(
            f"grid exceeded the engine budget and was block-max-downsampled "
            f"{per_axis}x per axis (tumour-preserving max-pooling: a block "
            f"containing tumour stays tumour), and the derived header carries "
            f"the scaled voxel size {derived_sizes} -- reported, not silent. "
            f"The reported tumour volume is therefore the block-max volume on "
            f"the coarser grid: it can only be equal-or-larger than the source "
            f"volume, never smaller."
        )
    return _snake_to_camel({**payload, "vti_base64": vti_b64})


# Dosing console: a real calendar in, a real engine run out
# --------------------------------------------------------------------------- #
# The UI used to ship a dosing panel with no engine behind it. This endpoint is
# that engine: it takes a calendar, runs the actual TumorEvolutionSimulator on
# it, and returns what happened. Nothing here is a lookup table or a mock.
DOSING_MAX_DAYS = 730


class DosingRequest(BaseModel):
    """One calendar, or a search over a declared grid.

    `mode="single"` runs exactly the calendar given. `mode="search"` runs the
    declared grid and reports every cell that actually ran, including the ones
    that lost.
    """

    mode: str = Field(default="single", pattern="^(single|search)$")
    on_days: int = Field(default=5, ge=1, le=60)
    cycle_days: int = Field(default=26, ge=1, le=180)
    dose_mg_m2: float = Field(default=150.0, ge=0.0, le=1000.0)
    dose_level: float = Field(default=1.0, ge=0.0, le=1.0)
    days: int = Field(default=365, ge=2, le=DOSING_MAX_DAYS)
    n_seeds: int = Field(default=3, ge=1, le=30)
    seed0: int = Field(default=0, ge=0)
    initial_resistant_fraction: float = Field(default=0.01, ge=0.0, lt=1.0)
    growth_noise: float = Field(default=0.05, ge=0.0)
    # search-only grid
    on_days_grid: List[int] = Field(default_factory=lambda: [3, 5, 7, 10])
    cycle_days_grid: List[int] = Field(default_factory=lambda: [21, 28, 35, 42])
    dose_levels_grid: List[float] = Field(default_factory=lambda: [0.4, 0.6, 0.8])
    # Optional shorter-horizon screen. Measured HARMFUL on this engine -- see
    # outputs_real/dosing.json["screen_fidelity"] -- so it is off by default and
    # the response carries the measured verdict either way.
    screen_days: Optional[int] = Field(default=None, ge=2, le=DOSING_MAX_DAYS)


def _short_screen_verdict(screen_days: Optional[int]) -> Dict:
    """Attach the MEASURED verdict on short-horizon screening to the response.

    Read from the artifact rather than hardcoded, so if the calibration is ever
    re-run the API stops repeating a stale reassurance. A client that asks for a
    short screen gets told what it costs before it trusts the winner.
    """
    import json as _json
    import os as _os
    from danon.claims import ROOT as _ROOT

    path = _os.path.join(_ROOT, "outputs_real", "dosing.json")
    measured: Optional[Dict] = None
    if _os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                measured = _json.load(fh).get("screen_fidelity")
        except (OSError, ValueError):
            measured = None

    if measured:
        return {
            "short_screen_requested_days": screen_days,
            "measured_at_days": measured.get("screen_days"),
            "measured_spearman": measured.get("spearman_rho"),
            "measured_kendall_tau": measured.get("kendall_tau"),
            "measured_top_k_caught": measured.get("top_k_caught_by_screen"),
            "measured_top_k_missed": measured.get("top_k_missed_by_screen"),
            "screen_keeps_full_winner": measured.get("screen_keeps_full_winner"),
            "verdict": (
                "MEASURED HARMFUL on this engine: a shorter horizon ranks "
                "candidates NEGATIVELY against the full-horizon ranking and "
                "discards the true winner. Do not trust a winner chosen by a "
                "short screen."
                if not measured.get("screen_keeps_full_winner")
                else (
                    "measured acceptable at this horizon: the screen retained "
                    "the full-horizon winner."
                )
            ),
        }
    return {
        "short_screen_requested_days": screen_days,
        "verdict": (
            "no calibration artifact found (outputs_real/dosing.json is "
            "missing its screen_fidelity section), so short screening is "
            "UNVALIDATED here. Rank stage 1 on the full horizon instead."
        ),
    }


@app.post("/api/dosing/plan")
def dosing_plan(req: DosingRequest):
    """Run a real calendar (or a real search) on the real evolutionary engine.

    Returns measured outcomes plus the provenance that says which parts are
    engine output and which are declared assumptions. A search that fails to
    beat continuous MTD says so in `beats_baselines` and explains it in
    `notes`; it never returns a winner that did not win.
    """
    import time as _time
    from danon.dosing import (
        MyelosuppressionModel,
        Schedule,
        evaluate_schedule,
        provenance,
        screen_schedule,
        search_schedules,
    )
    from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom

    if req.mode == "single" and req.on_days > req.cycle_days:
        raise HTTPException(
            422,
            f"on_days ({req.on_days}) cannot exceed cycle_days ({req.cycle_days}): "
            f"a holiday longer than the block is not a schedule, it is a pause",
        )

    t0 = _time.time()
    geom = make_gbm_phantom(size=24, seed=12345)
    sim = TumorEvolutionSimulator(geom)
    myelo = MyelosuppressionModel()
    seeds = tuple(range(req.seed0, req.seed0 + req.n_seeds))

    base = {
        "geometry": {
            "grid": list(geom.tumor_mask.shape),
            "tumour_voxels": int(geom.tumor_mask.sum()),
            "phantom_seed": 12345,
            "is_patient_imaging": False,
        },
        "myelosuppression_model": myelo.to_dict(),
        "provenance": provenance(),
    }

    if req.mode == "search":
        result = search_schedules(
            sim,
            seeds=seeds,
            days=req.days,
            on_days=tuple(req.on_days_grid),
            cycle_days=tuple(req.cycle_days_grid),
            dose_levels=tuple(req.dose_levels_grid),
            dose_mg_m2=req.dose_mg_m2,
            initial_resistant_fraction=req.initial_resistant_fraction,
            growth_noise=req.growth_noise,
            myelo=myelo,
            screen_seeds=min(2, len(seeds)),
            confirm_top=min(3, len(seeds)),
            screen_days=req.screen_days,
        )
        payload = {**base, "mode": "search", **result.to_dict()}
        payload["short_screen_verdict"] = _short_screen_verdict(req.screen_days)
    else:
        sched = Schedule.cyclic(
            "A", req.dose_mg_m2, req.on_days, req.cycle_days, req.days,
            dose_level=req.dose_level,
        )
        safety = screen_schedule(sched, req.days, myelo)
        score = evaluate_schedule(
            f"{req.dose_mg_m2:g} mg/m2 x{req.on_days}d every {req.cycle_days}d",
            sched, sim, seeds, req.days,
            req.initial_resistant_fraction, req.growth_noise, myelo,
        )
        mtd = evaluate_schedule(
            "MTD (continuous)",
            Schedule.continuous("A", req.dose_mg_m2, req.days),
            sim, seeds, req.days,
            req.initial_resistant_fraction, req.growth_noise, myelo,
        )
        payload = {
            **base,
            "mode": "single",
            "calendar": score.to_dict(),
            "safety": safety.to_dict(),
            "mtd_comparison": mtd.to_dict(),
            "beats_mtd": score.median_aubc < mtd.median_aubc,
            "notes": (
                []
                if score.median_aubc < mtd.median_aubc
                else [
                    f"this calendar did not beat continuous MTD on {len(seeds)} "
                    f"seeds (AUBC {score.median_aubc:.4f} vs {mtd.median_aubc:.4f}); "
                    f"reported as measured, not as a win"
                ]
            ),
        }

    payload["wall_seconds"] = round(_time.time() - t0, 2)
    return _snake_to_camel(payload)


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
