import os
import json
import logging
import numpy as np
import torch
from pipeline.config import PipelineConfig
from pipeline.models.architectures import (
    AAVTropismTransformer, LNPDeliveryMLP, ImmuneEscapeTransformer, AA_TO_IDX,
)
from pipeline.training.train_loops import ModelManager

logger = logging.getLogger(__name__)


class TrainedModelScorer:
    def __init__(self, config: PipelineConfig = None, checkpoint_dir: str = "./checkpoints"):
        self.config = config or PipelineConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_manager = ModelManager(checkpoint_dir)

        self.aav_model = None
        self.lnp_model = None
        self.immune_model = None

    def load_models(self, aav_checkpoint: str = None, lnp_checkpoint: str = None,
                    immune_checkpoint: str = None):
        logger.info("Loading trained models from %s", self.device)

        if aav_checkpoint:
            self.aav_model = self.model_manager.load_aav_model(aav_checkpoint)
        else:
            self.aav_model = self.model_manager.load_aav_model()

        if lnp_checkpoint:
            self.lnp_model = self.model_manager.load_lnp_model(lnp_checkpoint)
        else:
            self.lnp_model = self.model_manager.load_lnp_model()

        if immune_checkpoint:
            self.immune_model = self.model_manager.load_immune_model(immune_checkpoint)
        else:
            self.immune_model = self.model_manager.load_immune_model()

        logger.info("All trained models loaded successfully")

    def score_aav_candidates(self, candidates: list) -> list:
        if self.aav_model is None:
            self.load_models()

        self.aav_model.eval()
        batch_size = 64
        results = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            encoded = self._encode_sequences([c.sequence for c in batch])

            with torch.no_grad():
                predictions = self.aav_model(encoded.to(self.device))

            tissue_probs = torch.softmax(predictions["tissue_logits"], dim=-1)
            delivery_scores = predictions["delivery_score"].squeeze(-1)
            immune_scores = predictions["immune_score"].squeeze(-1)
            tissue_specificity = predictions["tissue_scores"]

            for j, candidate in enumerate(batch):
                candidate.tissue_score = tissue_probs[j].max().item()
                candidate.delivery_score = delivery_scores[j].item()
                candidate.immune_evasion_score = immune_scores[j].item()

                target_tissues = ["cardiac", "neuronal", "joint_cartilage"]
                target_idx = [self._tissue_to_idx(t) for t in target_tissues if t in self._tissue_names()]
                if target_idx:
                    candidate.tropism_score = tissue_specificity[j, target_idx].mean().item()
                else:
                    candidate.tropism_score = tissue_specificity[j].mean().item()

                avoid_tissues = ["hepatic", "renal"]
                avoid_idx = [self._tissue_to_idx(t) for t in avoid_tissues if t in self._tissue_names()]
                if avoid_idx:
                    liver_avoidance = 1.0 - tissue_specificity[j, avoid_idx].mean().item()
                else:
                    liver_avoidance = 0.5

                candidate.fitness = (
                    0.30 * candidate.immune_evasion_score +
                    0.25 * candidate.tropism_score +
                    0.25 * candidate.delivery_score +
                    0.20 * liver_avoidance
                )
                results.append(candidate)

        return results

    def score_lnp_candidates(self, candidates: list) -> list:
        if self.lnp_model is None:
            self.load_models()

        self.lnp_model.eval()
        batch_size = 64
        results = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            features = self._encode_lnp_features(batch)

            with torch.no_grad():
                predictions = self.lnp_model(features.to(self.device))

            delivery_scores = predictions["delivery_score"]
            organoid_counts = predictions["organoid_counts"]
            safety = predictions["safety"]

            for j, candidate in enumerate(batch):
                candidate.predicted_delivery = delivery_scores[j].item()
                candidate.predicted_organoid = {
                    "heart": organoid_counts[j, 0].item(),
                    "brain": organoid_counts[j, 1].item(),
                    "liver": organoid_counts[j, 2].item(),
                    "joint": organoid_counts[j, 3].item(),
                }
                candidate.predicted_safety = 1.0 - safety[j, 0].item()
                candidate.predicted_cytotoxicity = safety[j, 1].item()

                heart_score = organoid_counts[j, 0].item() / max(organoid_counts[j].sum().item(), 1)
                liver_penalty = organoid_counts[j, 2].item() / max(organoid_counts[j].sum().item(), 1)

                candidate.fitness = (
                    0.35 * candidate.predicted_delivery +
                    0.25 * heart_score +
                    0.20 * candidate.predicted_safety +
                    0.20 * (1.0 - liver_penalty)
                )
                results.append(candidate)

        return results

    def score_immune_escape(self, candidates: list) -> list:
        if self.immune_model is None:
            self.load_models()

        self.immune_model.eval()
        batch_size = 64
        results = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            encoded = self._encode_sequences([c.sequence for c in batch])

            with torch.no_grad():
                predictions = self.immune_model(encoded.to(self.device))

            escape_scores = predictions["escape_scores"]
            binding_energies = predictions["binding_energies"]
            total_escape = predictions["total_escape"]
            resistance = predictions["resistance"]

            for j, candidate in enumerate(batch):
                candidate.antibody_escape_scores = {
                    "AAV2_Ab4": escape_scores[j, 0].item(),
                    "AAV2_Ab58": escape_scores[j, 1].item(),
                    "AAV8_Ab1": escape_scores[j, 2].item(),
                    "AAV9_Ab3": escape_scores[j, 3].item(),
                    "human_IgG": escape_scores[j, 4].item(),
                    "human_IgM": escape_scores[j, 5].item(),
                    "anti-AAV9": escape_scores[j, 6].item(),
                }
                candidate.total_immune_escape = total_escape[j].item()
                candidate.neutralization_resistance = resistance[j].item()

                candidate.immune_evasion_score = (
                    0.4 * total_escape[j].item() +
                    0.3 * resistance[j].item() +
                    0.3 * escape_scores[j].mean().item()
                )
                results.append(candidate)

        return results

    def _encode_sequences(self, sequences: list, max_len: int = 750) -> torch.Tensor:
        encoded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:max_len]):
                if aa in AA_TO_IDX:
                    encoded[i, j] = AA_TO_IDX[aa]
        return encoded

    def _encode_lnp_features(self, candidates: list) -> torch.Tensor:
        lipid_map = {
            "DLin-MC3-DMA": 0, "SM-102": 1, "ALC-0315": 2,
            "DODAP": 3, "DLin-DMA": 4, "cKK-E11": 5,
        }
        peg_map = {"DMG-PEG2000": 0, "DSPC-PEG2000": 1, "DSPE-PEG2000": 2}
        helper_map = {"DSPC": 0, "DPPC": 1, "DOPE": 2, "POPC": 3}

        features = []
        for c in candidates:
            features.append([
                lipid_map.get(c.ionizable_lipid, 0),
                peg_map.get(c.peg_lipid, 0),
                helper_map.get(c.helper_lipid, 0),
                c.ionizable_frac,
                c.peg_frac,
                c.cholesterol_frac,
                c.pka,
                c.tail_length / 22.0,
                c.unsaturation / 5.0,
            ])
        return torch.tensor(features, dtype=torch.float32)

    def _tissue_names(self):
        return ["cardiac", "neuronal", "joint_cartilage", "skeletal_muscle",
                "hepatic", "renal", "pulmonary", "adipose"]

    def _tissue_to_idx(self, tissue: str):
        names = self._tissue_names()
        return names.index(tissue) if tissue in names else 0


class SpatialTranscriptomicsIntegrator:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir

    def process_spatial_data(self, spatial_data_path: str) -> dict:
        import json
        with open(spatial_data_path) as f:
            data = json.load(f)

        tissue_maps = {}
        for sample in data:
            tissue = sample["tissue"]
            if tissue not in tissue_maps:
                tissue_maps[tissue] = {
                    "samples": 0,
                    "total_cells": 0,
                    "avg_aging_index": 0,
                    "cell_type_distribution": {},
                    "hotspot_regions": [],
                }

            tissue_maps[tissue]["samples"] += 1
            tissue_maps[tissue]["total_cells"] += sample["num_cells"]
            tissue_maps[tissue]["avg_aging_index"] += sample["tissue_aging_index"]

            for cell in sample["cells"]:
                ct = cell["cell_type"]
                if ct not in tissue_maps[tissue]["cell_type_distribution"]:
                    tissue_maps[tissue]["cell_type_distribution"][ct] = 0
                tissue_maps[tissue]["cell_type_distribution"][ct] += 1

                if cell["aging_score"] > 0.7:
                    tissue_maps[tissue]["hotspot_regions"].append({
                        "x": cell["x"], "y": cell["y"], "z": cell["z"],
                        "aging_score": cell["aging_score"],
                        "senescence_level": cell["senescence_marker_level"],
                    })

        for tissue in tissue_maps:
            n = tissue_maps[tissue]["samples"]
            tissue_maps[tissue]["avg_aging_index"] /= max(n, 1)

        return tissue_maps

    def identify_target_regions(self, tissue_maps: dict) -> list:
        targets = []
        for tissue, info in tissue_maps.items():
            if info["avg_aging_index"] > 0.5:
                hotspots = sorted(
                    info["hotspot_regions"],
                    key=lambda x: x["aging_score"],
                    reverse=True
                )[:10]

                targets.append({
                    "tissue": tissue,
                    "priority": "high" if info["avg_aging_index"] > 0.7 else "medium",
                    "aging_index": info["avg_aging_index"],
                    "top_hotspots": hotspots,
                    "target_cell_types": [
                        ct for ct, count in info["cell_type_distribution"].items()
                        if count / max(info["total_cells"], 1) > 0.1
                    ],
                })
        return sorted(targets, key=lambda x: x["aging_index"], reverse=True)
