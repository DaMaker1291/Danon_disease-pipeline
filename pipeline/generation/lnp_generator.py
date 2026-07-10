import logging
import numpy as np
import math
from typing import Iterator
from dataclasses import dataclass
from pipeline.config import GenerationConfig

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. Using fallback molecular representation.")

IONIZABLE_LIPIDS = {
    "DLin-MC3-DMA": {"tails": 18, "unsaturation": 2, "pka": 6.44, "mw": 918.44,
                      "smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)N[C@H](C[N+](C)(C)C)OC(=O)CCCCCCC/C=C\\CCCCCCCC"},
    "SM-102": {"tails": 14, "unsaturation": 1, "pka": 6.27, "mw": 654.02,
               "smiles": "CCCCCCCCCCCC(=O)OC[C@H](C[N+](C)(C)C)OC(=O)CCCCCCC/C=C\\CCCCCCCC"},
    "ALC-0315": {"tails": 18, "unsaturation": 2, "pka": 6.09, "mw": 766.18,
                 "smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](C[N+](C)(C)C)OC(=O)CCCCCCC/C=C\\CCCCCCCC"},
    "DODAP": {"tails": 18, "unsaturation": 1, "pka": 6.44, "mw": 591.01,
              "smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](C[N+](C)(C)C)OC(=O)CCCCCCCC/C=C\\CCCCCCCC"},
    "DLin-DMA": {"tails": 18, "unsaturation": 2, "pka": 6.84, "mw": 650.15,
                 "smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](C[N+](C)(C)C)OC(=O)CCCCCCCC/C=C\\CCCCCCCC"},
    "cKK-E11": {"tails": 14, "unsaturation": 1, "pka": 6.07, "mw": 610.00,
                "smiles": "CCCCCCCC(=O)OC[C@H](C[N+](C)(C)C)OC(=O)CCCCCCCC/C=C\\CCCCCCCC"},
}

PEG_LIPIDS = {
    "DMG-PEG2000": {"peg_mw": 2000, "tail": "dimyristoyl", "mol_frac": 0.015},
    "DSPC-PEG2000": {"peg_mw": 2000, "tail": "distearoyl", "mol_frac": 0.010},
    "DSPE-PEG2000": {"peg_mw": 2000, "tail": "distearoyl", "mol_frac": 0.010},
    "DSPC-PEG5000": {"peg_mw": 5000, "tail": "distearoyl", "mol_frac": 0.008},
}

HELPER_LIPIDS = {
    "DSPC": {"tail": "distearoyl", "phase": "gel", "t_m": 55.0,
             "smiles": "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCC"},
    "DPPC": {"tail": "dipalmitoyl", "phase": "gel", "t_m": 41.0},
    "DOPE": {"tail": "dioleoyl", "phase": "hexagonal", "t_m": -16.0},
    "POPC": {"tail": "palmitoyloleoyl", "phase": "liquid_crystal", "t_m": -2.0},
}

CHOLESTEROL = {"role": "stabilizer", "mol_frac_range": (0.25, 0.40),
               "smiles": "C1CCC2C3CCC4CC(O)CCC4(C)C3CCC2(C)C1(C)C(C)CCCC(C)C(O)C"}

APOE_BINDING_SITES = {
    "receptor_binding_domain": {"residues": [130, 131, 132, 133, 134, 135, 136],
                                "hydrophobic_preference": 0.7, "charge_preference": -0.3},
    "heparan_sulfate_site": {"residues": [34, 35, 36, 37, 38],
                             "hydrophobic_preference": 0.3, "charge_preference": 0.8},
}


@dataclass
class LNPCandidate:
    candidate_id: int
    ionizable_lipid: str
    peg_lipid: str
    helper_lipid: str
    ionizable_frac: float
    peg_frac: float
    helper_frac: float
    cholesterol_frac: float
    tail_length: int
    unsaturation: int
    pka: float
    predicted_transfection: float = 0.0
    predicted_endosomal_escape: float = 0.0
    predicted_stability: float = 0.0
    apoe_binding_score: float = 0.0
    particle_size_nm: float = 0.0
    md_stability_score: float = 0.0
    fitness: float = 0.0
    smiles: str = ""

    def composition_dict(self) -> dict:
        return {
            "ionizable_lipid": self.ionizable_lipid, "peg_lipid": self.peg_lipid,
            "helper_lipid": self.helper_lipid, "ionizable_frac": self.ionizable_frac,
            "peg_frac": self.peg_frac, "helper_frac": self.helper_frac,
            "cholesterol_frac": self.cholesterol_frac, "tail_length": self.tail_length,
            "unsaturation": self.unsaturation, "pka": self.pka,
        }


class MDSimulationProxy:
    def __init__(self):
        self.temperature = 310.15
        self.kb = 0.0019872041

    def estimate_free_energy(self, candidate: LNPCandidate) -> float:
        tail_energy = -0.5 * candidate.tail_length * (1.0 + 0.3 * candidate.unsaturation)
        pka_penalty = -2.0 * ((candidate.pka - 6.35) / 0.3) ** 2
        cholesterol_stab = 3.0 * candidate.cholesterol_frac
        peg_steric = -5.0 * candidate.peg_frac
        ion_interaction = 2.0 * candidate.ionizable_frac

        total_energy = tail_energy + pka_penalty + cholesterol_stab + peg_steric + ion_interaction
        boltzmann_factor = np.exp(-total_energy / (self.kb * self.temperature))
        return float(np.clip(boltzmann_factor / (1 + boltzmann_factor), 0, 1))

    def predict_particle_size(self, candidate: LNPCandidate) -> float:
        base_size = 50.0
        peg_effect = candidate.peg_frac * 2000
        tail_effect = candidate.tail_length * 1.5
        cholesterol_effect = candidate.cholesterol_frac * (-30.0)
        ion_effect = candidate.ionizable_frac * (-20.0)

        size = base_size + peg_effect + tail_effect + cholesterol_effect + ion_effect
        return float(np.clip(size, 20, 200))


class ApoEInteractionModel:
    def __init__(self):
        self.apoe_sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELQQQTETLQVQVKAVETELQELQVQVQVEVKVEVKVEVKVEVEVKVEVKVEVKVEVK"
        self.binding_sites = APOE_BINDING_SITES

    def predict_apoe_binding(self, candidate: LNPCandidate) -> float:
        pka_score = np.exp(-0.5 * ((candidate.pka - 6.35) / 0.2) ** 2)

        tail_hydrophobicity = candidate.tail_length * 0.05 * (1 + 0.2 * candidate.unsaturation)
        hydro_score = np.exp(-0.5 * ((tail_hydrophobicity - 0.8) / 0.2) ** 2)

        peg_shielding = 1.0 - min(1.0, candidate.peg_frac * 50)

        ion_charge = 1.0 / (1.0 + np.exp(-(candidate.pka - 6.35) * 10))
        charge_score = np.exp(-0.5 * ((ion_charge - 0.6) / 0.2) ** 2)

        binding = 0.30 * pka_score + 0.25 * hydro_score + 0.25 * peg_shielding + 0.20 * charge_score
        return float(np.clip(binding, 0, 1))


class LNPGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.component_config = config.lnp_components
        self.rng = np.random.RandomState(42)
        self.md_proxy = MDSimulationProxy()
        self.apoe_model = ApoEInteractionModel()

    def generate_candidates(self, batch_id: int, batch_size: int) -> list[LNPCandidate]:
        candidates = []
        rng = np.random.RandomState(batch_id)
        for i in range(batch_size):
            global_id = batch_id * batch_size + i
            candidate = self._generate_single(rng, global_id)
            candidates.append(candidate)
        return candidates

    def _generate_single(self, rng: np.random.RandomState, global_id: int) -> LNPCandidate:
        ion_name = rng.choice(list(IONIZABLE_LIPIDS.keys()))
        ion_info = IONIZABLE_LIPIDS[ion_name]
        peg_name = rng.choice(list(PEG_LIPIDS.keys()))
        helper_name = rng.choice(list(HELPER_LIPIDS.keys()))

        ion_frac = rng.uniform(0.30, 0.50)
        peg_frac = rng.uniform(0.005, 0.025)
        cholesterol_frac = rng.uniform(0.25, 0.40)
        helper_frac = 1.0 - ion_frac - peg_frac - cholesterol_frac

        tail_length = int(ion_info["tails"] + rng.randint(-4, 5))
        tail_length = max(8, min(24, tail_length))
        unsaturation = int(ion_info["unsaturation"] + rng.randint(-1, 2))
        unsaturation = max(0, min(6, unsaturation))
        pka = ion_info["pka"] + rng.normal(0, 0.3)
        pka = np.clip(pka, 5.5, 7.0)

        smiles = ion_info.get("smiles", "")

        return LNPCandidate(
            candidate_id=global_id, ionizable_lipid=ion_name,
            peg_lipid=peg_name, helper_lipid=helper_name,
            ionizable_frac=ion_frac, peg_frac=peg_frac,
            helper_frac=helper_frac, cholesterol_frac=cholesterol_frac,
            tail_length=tail_length, unsaturation=unsaturation,
            pka=pka, smiles=smiles,
        )

    def score_candidates(self, candidates: list[LNPCandidate]) -> list[LNPCandidate]:
        for c in candidates:
            c.predicted_transfection = self._predict_transfection(c)
            c.predicted_endosomal_escape = self._predict_endosomal_escape(c)
            c.predicted_stability = self._predict_stability(c)
            c.apoe_binding_score = self.apoe_model.predict_apoe_binding(c)
            c.particle_size_nm = self.md_proxy.predict_particle_size(c)
            c.md_stability_score = self.md_proxy.estimate_free_energy(c)

            size_penalty = np.exp(-0.5 * ((c.particle_size_nm - 80) / 30) ** 2)

            c.fitness = (
                0.25 * c.predicted_transfection +
                0.20 * c.predicted_endosomal_escape +
                0.15 * c.predicted_stability +
                0.15 * c.apoe_binding_score +
                0.10 * size_penalty +
                0.15 * c.md_stability_score
            )
        return candidates

    def _predict_transfection(self, c: LNPCandidate) -> float:
        ion_score = np.exp(-0.5 * ((c.ionizable_frac - 0.40) / 0.08) ** 2)
        pka_score = np.exp(-0.5 * ((c.pka - 6.3) / 0.2) ** 2)
        tail_score = np.exp(-0.5 * ((c.tail_length - 16) / 4) ** 2)
        peg_penalty = 1.0 - 3.0 * c.peg_frac
        return float(np.clip(
            0.3 * ion_score + 0.3 * pka_score + 0.2 * tail_score + 0.2 * peg_penalty, 0, 1))

    def _predict_endosomal_escape(self, c: LNPCandidate) -> float:
        pka_escape = np.exp(-0.5 * ((c.pka - 6.4) / 0.25) ** 2)
        unsat_escape = np.exp(-0.5 * ((c.unsaturation - 2) / 1.5) ** 2)
        helper_escape = 1.0 if c.helper_lipid == "DOPE" else 0.6
        return float(np.clip(
            0.4 * pka_escape + 0.3 * unsat_escape + 0.3 * helper_escape, 0, 1))

    def _predict_stability(self, c: LNPCandidate) -> float:
        chol_score = np.exp(-0.5 * ((c.cholesterol_frac - 0.35) / 0.05) ** 2)
        helper_score = 1.0 if c.helper_lipid in ["DSPC", "DPPC"] else 0.7
        tail_stab = np.exp(-0.5 * ((c.tail_length - 18) / 3) ** 2)
        return float(np.clip(
            0.4 * chol_score + 0.3 * helper_score + 0.3 * tail_stab, 0, 1))

    def stream_candidates(self, total: int, batch_size: int) -> Iterator[list[LNPCandidate]]:
        num_batches = (total + batch_size - 1) // batch_size
        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, total - batch_id * batch_size)
            candidates = self.generate_candidates(batch_id, current_batch_size)
            scored = self.score_candidates(candidates)
            yield scored
