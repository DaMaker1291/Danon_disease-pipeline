import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data")


class SyntheticDataGenerator:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_aav_tropism_data(self, num_samples: int = 100000) -> str:
        rng = np.random.RandomState(42)
        AA = "ACDEFGHIKLMNPQRSTVWY"

        AAV_SEROTYPES = {
            "AAV1": {"cardiac": 0.85, "skeletal_muscle": 0.90, "neuronal": 0.30},
            "AAV2": {"cardiac": 0.40, "retinal": 0.80, "liver": 0.35},
            "AAV5": {"cardiac": 0.75, "neuronal": 0.85, "lung": 0.70},
            "AAV8": {"liver": 0.95, "cardiac": 0.60, "kidney": 0.55},
            "AAV9": {"cardiac": 0.92, "neuronal": 0.88, "skeletal_muscle": 0.85},
            "AAVrh10": {"neuronal": 0.90, "cardiac": 0.70, "lung": 0.65},
            "AAV-PHP.eB": {"neuronal": 0.98, "brain": 0.95, "spinal_cord": 0.88},
            "AAVrh74": {"skeletal_muscle": 0.92, "cardiac": 0.75, "liver": 0.60},
        }

        TISSUES = ["cardiac", "neuronal", "joint_cartilage", "skeletal_muscle",
                   "hepatic", "renal", "pulmonary", "adipose"]

        data = []
        for i in range(num_samples):
            serotype = rng.choice(list(AAV_SEROTYPES.keys()))
            tropism_prefs = AAV_SEROTYPES[serotype]

            seq_len = rng.randint(700, 750)
            sequence = "".join(rng.choice(list(AA), size=seq_len))

            mutations = []
            n_mutations = rng.randint(0, 8)
            for _ in range(n_mutations):
                pos = rng.randint(263, min(732, seq_len - 1))
                orig_aa = sequence[pos]
                new_aa = rng.choice([aa for aa in AA if aa != orig_aa])
                mutations.append((pos, orig_aa, new_aa))
                sequence = sequence[:pos] + new_aa + sequence[pos + 1:]

            tissue_scores = {}
            for tissue in TISSUES:
                if tissue in tropism_prefs:
                    base = tropism_prefs[tissue]
                else:
                    base = rng.beta(2, 5)
                noise = rng.normal(0, 0.08)
                tissue_scores[tissue] = float(np.clip(base + noise, 0, 1))

            primary_tissue = max(tissue_scores, key=tissue_scores.get)

            delivery_eff = float(np.clip(
                tissue_scores[primary_tissue] * 0.8 + rng.normal(0, 0.1), 0, 1
            ))

            immune_escape = float(np.clip(
                0.3 + 0.4 * (len(mutations) / 10) + rng.normal(0, 0.15), 0, 1
            ))

            stability = float(np.clip(
                0.5 + 0.3 * np.exp(-0.5 * ((len(sequence) - 730) / 50) ** 2) +
                rng.normal(0, 0.1), 0, 1
            ))

            data.append({
                "sequence": sequence,
                "tissue_scores": tissue_scores,
                "tropism_target": primary_tissue,
                "delivery_efficiency": delivery_eff,
                "immune_escape_score": immune_escape,
                "stability_score": stability,
                "mutations": mutations,
                "serotype": serotype,
            })

        filepath = os.path.join(self.data_dir, "synthetic_aav_tropism.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        logger.info("Generated %d AAV tropism samples -> %s", num_samples, filepath)
        return filepath

    def generate_lnp_delivery_data(self, num_samples: int = 100000) -> str:
        rng = np.random.RandomState(42)
        LIPIDS = ["DLin-MC3-DMA", "SM-102", "ALC-0315", "DODAP", "DLin-DMA", "cKK-E11"]
        PEG_LIPIDS = ["DMG-PEG2000", "DSPC-PEG2000", "DSPE-PEG2000"]
        HELPER_LIPIDS = ["DSPC", "DPPC", "DOPE", "POPC"]

        KNOWN_EFFICIENCIES = {
            "DLin-MC3-DMA": {"liver": 0.92, "pKa_opt": 6.44},
            "SM-102": {"liver": 0.88, "pKa_opt": 6.29},
            "ALC-0315": {"liver": 0.90, "pKa_opt": 6.09},
            "DODAP": {"lung": 0.75, "pKa_opt": 6.50},
            "DLin-DMA": {"liver": 0.70, "pKa_opt": 6.30},
            "cKK-E11": {"spleen": 0.85, "pKa_opt": 6.24},
        }

        data = []
        for i in range(num_samples):
            ion_lipid = rng.choice(LIPIDS)
            peg_lipid = rng.choice(PEG_LIPIDS)
            helper_lipid = rng.choice(HELPER_LIPIDS)

            ion_frac = rng.uniform(0.30, 0.50)
            peg_frac = rng.uniform(0.005, 0.025)
            chol_frac = rng.uniform(0.25, 0.40)
            helper_frac = 1.0 - ion_frac - peg_frac - chol_frac

            pka = rng.normal(KNOWN_EFFICIENCIES[ion_lipid]["pKa_opt"], 0.2)
            tail_length = rng.randint(12, 22)
            unsaturation = rng.randint(0, 5)

            efficiency = float(np.clip(
                0.3 * np.exp(-0.5 * ((pka - KNOWN_EFFICIENCIES[ion_lipid]["pKa_opt"]) / 0.15) ** 2) +
                0.2 * np.exp(-0.5 * ((ion_frac - 0.40) / 0.06) ** 2) +
                0.15 * (1.0 if helper_lipid == "DOPE" else 0.6) +
                0.15 * np.exp(-0.5 * ((tail_length - 16) / 3) ** 2) +
                0.1 * (1.0 if peg_lipid == "DMG-PEG2000" else 0.7) +
                0.1 * rng.random(), 0, 1
            ))

            organoid_counts = {}
            for organ in ["heart_organoid", "brain_organoid", "liver_organoid", "joint_organoid"]:
                organoid_counts[organ] = int(rng.poisson(efficiency * 1000))

            particle_size = float(np.clip(
                60 + 20 * np.exp(-0.5 * ((peg_frac - 0.015) / 0.008) ** 2) +
                rng.normal(0, 5), 40, 120
            ))

            pdi = float(np.clip(
                0.05 + 0.1 * (1.0 - efficiency) + rng.normal(0, 0.02), 0.01, 0.3
            ))

            data.append({
                "ionizable_lipid": ion_lipid, "peg_lipid": peg_lipid,
                "helper_lipid": helper_lipid, "ionizable_frac": float(ion_frac),
                "peg_frac": float(peg_frac), "cholesterol_frac": float(chol_frac),
                "helper_frac": float(helper_frac),
                "pka": float(pka), "tail_length": int(tail_length),
                "unsaturation": int(unsaturation), "delivery_efficiency": efficiency,
                "organoid_barcode_counts": organoid_counts,
                "particle_size_nm": particle_size,
                "polydispersity_index": pdi,
                "immune_activation": float(rng.beta(2, 5)),
                "cytotoxicity": float(rng.beta(1, 8)),
            })

        filepath = os.path.join(self.data_dir, "synthetic_lnp_delivery.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        logger.info("Generated %d LNP delivery samples -> %s", num_samples, filepath)
        return filepath

    def generate_immune_escape_data(self, num_samples: int = 50000) -> str:
        rng = np.random.RandomState(42)
        AA = "ACDEFGHIKLMNPQRSTVWY"

        ANTIBODIES = {
            "AAV2_Ab4": {"epitope_region": (263, 450), "escape_difficulty": 0.7},
            "AAV2_Ab58": {"epitope_region": (450, 580), "escape_difficulty": 0.6},
            "AAV8_Ab1": {"epitope_region": (200, 380), "escape_difficulty": 0.8},
            "AAV9_Ab3": {"epitope_region": (500, 650), "escape_difficulty": 0.5},
            "human_IgG_pool": {"epitope_region": (100, 700), "escape_difficulty": 0.3},
            "human_IgM_pool": {"epitope_region": (100, 700), "escape_difficulty": 0.2},
            "anti-AAV9_serum": {"epitope_region": (400, 600), "escape_difficulty": 0.9},
        }

        data = []
        for i in range(num_samples):
            seq_len = rng.randint(700, 750)
            sequence = "".join(rng.choice(list(AA), size=seq_len))

            mutations = []
            n_mutations = rng.randint(0, 12)
            for _ in range(n_mutations):
                pos = rng.randint(263, min(732, seq_len - 1))
                orig_aa = sequence[pos]
                new_aa = rng.choice([aa for aa in AA if aa != orig_aa])
                mutations.append((pos, orig_aa, new_aa))
                sequence = sequence[:pos] + new_aa + sequence[pos + 1:]

            antibody_scores = {}
            for ab_name, ab_info in ANTIBODIES.items():
                ep_start, ep_end = ab_info["epitope_region"]
                mutations_in_epitope = sum(
                    1 for pos, _, _ in mutations if ep_start <= pos <= ep_end
                )

                base_binding = rng.beta(2, 3)
                escape_boost = mutations_in_epitope * 0.08
                escape_score = float(np.clip(base_binding + escape_boost, 0, 1))

                antibody_scores[ab_name] = {
                    "binding_energy": float(-10 * (1 - escape_score) + rng.normal(0, 0.5)),
                    "escape_score": escape_score,
                    "mutations_in_epitope": mutations_in_epitope,
                }

            total_escape = float(np.mean([v["escape_score"] for v in antibody_scores.values()]))
            neutralization_resistance = float(np.clip(
                0.3 + 0.5 * total_escape + 0.2 * (n_mutations / 12) + rng.normal(0, 0.1), 0, 1
            ))

            data.append({
                "sequence": sequence,
                "antibody_responses": antibody_scores,
                "total_escape_score": total_escape,
                "neutralization_resistance": neutralization_resistance,
                "mutations": mutations,
                "n_mutations": n_mutations,
            })

        filepath = os.path.join(self.data_dir, "synthetic_immune_escape.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        logger.info("Generated %d immune escape samples -> %s", num_samples, filepath)
        return filepath

    def generate_spatial_transcriptomics(self, num_samples: int = 10000) -> str:
        rng = np.random.RandomState(42)
        CELLS = ["cardiomyocyte", "fibroblast", "endothelial", "macrophage",
                 "smooth_muscle", "adipocyte", "pericyte", "neuron"]
        TISSUES = ["heart", "brain", "joint", "liver"]

        data = []
        for i in range(num_samples):
            tissue = rng.choice(TISSUES)
            num_cells = rng.randint(1000, 5000)
            cells = []
            for j in range(num_cells):
                cells.append({
                    "cell_id": j,
                    "cell_type": rng.choice(CELLS).tolist(),
                    "x": float(rng.uniform(0, 1000)),
                    "y": float(rng.uniform(0, 1000)),
                    "z": float(rng.uniform(0, 100)),
                    "expression_vector": rng.exponential(1.0, size=100).tolist(),
                    "aging_score": float(rng.beta(2, 3)),
                    "senescence_marker_level": float(rng.beta(1, 5)),
                })

            data.append({
                "tissue": tissue, "num_cells": num_cells, "cells": cells,
                "tissue_aging_index": float(rng.beta(3, 2)),
                "collagen_crosslink_density": float(rng.beta(2, 3)),
            })

        filepath = os.path.join(self.data_dir, "synthetic_spatial_transcriptomics.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        logger.info("Generated %d spatial transcriptomics samples -> %s", num_samples, filepath)
        return filepath

    def generate_all_synthetic(self) -> dict:
        return {
            "aav_tropism": self.generate_aav_tropism_data(),
            "lnp_delivery": self.generate_lnp_delivery_data(),
            "immune_escape": self.generate_immune_escape_data(),
            "spatial_transcriptomics": self.generate_spatial_transcriptomics(),
        }
