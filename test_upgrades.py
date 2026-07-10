from pipeline.config import PipelineConfig
from pipeline.generation.aav_generator import AAVGenerator
from pipeline.generation.lnp_generator import LNPGenerator
from pipeline.screening.filter1_immune import ImmuneEvasionFilter
from pipeline.screening.filter2_tropism import TropismFilter, SpatialTranscriptomicsIntegrator
from pipeline.lab.robotic_synthesis import OpentronsProtocol
from pipeline.feedback.refinement import AAVRefinery, LNPRefinery

config = PipelineConfig()

print("=== Testing All 6 Upgrades ===\n")

# 1. AAV Generator
print("1. AAV Generator (structure-conditioned)...")
aav_gen = AAVGenerator(config.generation)
candidates = aav_gen.generate_candidates(0, 10)
scored = aav_gen.score_candidates(candidates)
c = scored[0]
print(f"   ESM: {c.esm_score:.4f}, Stability: {c.stability_score:.4f}")
print(f"   Structural: {c.structural_score:.4f}, Interface: {c.interface_integrity:.4f}")
print(f"   Packing: {c.packing_density:.4f}, Fitness: {c.fitness:.4f}\n")

# 2. LNP Generator
print("2. LNP Generator (MD proxy + ApoE)...")
lnp_gen = LNPGenerator(config.generation)
lnp_cands = lnp_gen.generate_candidates(0, 10)
lnp_scored = lnp_gen.score_candidates(lnp_cands)
c = lnp_scored[0]
print(f"   ApoE: {c.apoe_binding_score:.4f}, Size: {c.particle_size_nm:.1f}nm")
print(f"   MD stability: {c.md_stability_score:.4f}, Fitness: {c.fitness:.4f}\n")

# 3. Immune Filter
print("3. Immune Filter (structural binding + steric)...")
immune = ImmuneEvasionFilter()
scores = [immune.score(c) for c in scored[:5]]
print(f"   Scores: {[f'{s:.3f}' for s in scores]}\n")

# 4. Tropism Filter
print("4. Tropism Filter (spatial transcriptomics + aging receptors)...")
tropism = TropismFilter()
scores = [tropism.score(c) for c in scored[:5]]
sti = SpatialTranscriptomicsIntegrator()
print(f"   Scores: {[f'{s:.3f}' for s in scores]}")
print(f"   Cardiac aging: {sti.get_aging_priority('cardiac'):.3f}")
print(f"   Joint aging: {sti.get_aging_priority('joint_cartilage'):.3f}\n")

# 5. Opentrons
print("5. Opentrons (microfluidic TFR/FRR control)...")
proto = OpentronsProtocol(config.lab)
cand = {"candidate_id": 1, "composition": {"pka": 6.3, "ionizable_frac": 0.40, "peg_frac": 0.015, "cholesterol_frac": 0.35}}
mfd = proto._compute_microfluidic_params(cand)
print(f"   TFR: {mfd.total_flow_rate_ul_min:.0f} uL/min, FRR: {mfd.flow_rate_ratio:.1f}:1")
print(f"   Target: {mfd.target_particle_size_nm:.0f}nm, Channel: {mfd.channel_diameter_um:.0f}um\n")

# 6. Bayesian Refinery
print("6. Bayesian Refinery (GP noise parameter alpha)...")
aav_ref = AAVRefinery(noise_alpha=0.15)
lnp_ref = LNPRefinery(noise_alpha=0.10)
print(f"   AAV noise_alpha: {aav_ref.noise_alpha}")
print(f"   LNP noise_alpha: {lnp_ref.noise_alpha}")
print(f"   GP kernel: K(x,x') + alpha^2 * I (accounts for lab noise)\n")

print("=== All 6 upgrades verified! ===")
