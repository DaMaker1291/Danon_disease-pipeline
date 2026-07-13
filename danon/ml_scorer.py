"""
ML Scorer: loads fine-tuned Kaggle checkpoints and scores candidates
using the Danon-optimized models (AAV Tropism Transformer, LNP MLP, Immune Transformer).
"""
import os
import math
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints_danon")

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class AAVTropismTransformer(nn.Module):
    def __init__(self, vocab_size=20, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, num_tissues=8, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.delivery_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.immune_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.tissue_score_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_tissues), nn.Sigmoid(),
        )

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled = encoded.mean(dim=1)
        return {
            "delivery_score": self.delivery_head(pooled),
            "immune_score": self.immune_head(pooled),
            "tissue_scores": self.tissue_score_head(pooled),
            "encoded": pooled,
        }


class LNPDeliveryMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4), nn.ReLU(),
        )
        self.delivery_head = nn.Linear(hidden_dim // 4, 1)
        self.cardiac_head = nn.Linear(hidden_dim // 4, 1)
        self.hepatic_head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x):
        h = self.network(x)
        return {
            "delivery_score": torch.sigmoid(self.delivery_head(h)).squeeze(-1),
            "cardiac_delivery": torch.sigmoid(self.cardiac_head(h)).squeeze(-1),
            "hepatic_avoidance": torch.sigmoid(self.hepatic_head(h)).squeeze(-1),
        }


class ImmuneEscapeTransformer(nn.Module):
    def __init__(self, vocab_size=20, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, num_antibodies=7, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.total_escape_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.resistance_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)
        return {
            "total_escape": self.total_escape_head(pooled).squeeze(-1),
            "resistance": self.resistance_head(pooled).squeeze(-1),
        }


class MLScorer:
    def __init__(self, checkpoint_dir: str = CKPT_DIR, device: str = None):
        self.ckpt_dir = checkpoint_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.aav_model = None
        self.lnp_model = None
        self.immune_model = None
        self._load_models()

    def _load_models(self):
        aav_path = os.path.join(self.ckpt_dir, "aav_danon_best.pt")
        lnp_path = os.path.join(self.ckpt_dir, "lnp_danon_best.pt")
        immune_path = os.path.join(self.ckpt_dir, "immune_danon_best.pt")

        if os.path.exists(aav_path):
            self.aav_model = AAVTropismTransformer().to(self.device)
            self.aav_model.load_state_dict(torch.load(aav_path, map_location=self.device, weights_only=True), strict=False)
            self.aav_model.eval()
            logger.info("  Loaded AAV model: %s", aav_path)
        else:
            logger.warning("  AAV checkpoint not found: %s", aav_path)

        if os.path.exists(lnp_path):
            self.lnp_model = LNPDeliveryMLP().to(self.device)
            self.lnp_model.load_state_dict(torch.load(lnp_path, map_location=self.device, weights_only=True), strict=False)
            self.lnp_model.eval()
            logger.info("  Loaded LNP model: %s", lnp_path)
        else:
            logger.warning("  LNP checkpoint not found: %s", lnp_path)

        if os.path.exists(immune_path):
            self.immune_model = ImmuneEscapeTransformer().to(self.device)
            self.immune_model.load_state_dict(torch.load(immune_path, map_location=self.device, weights_only=True), strict=False)
            self.immune_model.eval()
            logger.info("  Loaded Immune model: %s", immune_path)
        else:
            logger.warning("  Immune checkpoint not found: %s", immune_path)

    def encode_seq(self, seq: str, max_len: int = 50):
        encoded = torch.zeros(max_len, dtype=torch.long)
        for i, aa in enumerate(seq[:max_len]):
            if aa in AA_TO_IDX:
                encoded[i] = AA_TO_IDX[aa]
        return encoded.unsqueeze(0)

    def _enable_mc_dropout(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _mc_forward_pass(self, model: nn.Module, input_tensor: torch.Tensor):
        self._enable_mc_dropout(model)
        with torch.no_grad():
            return model(input_tensor)

    def _mc_predict(self, model: nn.Module, input_tensor: torch.Tensor,
                     forward_fn, n_passes: int = 50) -> Tuple[Dict[str, float], Dict[str, float]]:
        all_outputs = []
        for _ in range(n_passes):
            out = self._mc_forward_pass(model, input_tensor)
            all_outputs.append(forward_fn(out))
        means = {}
        stds = {}
        keys = all_outputs[0].keys()
        for k in keys:
            vals = np.array([o[k] for o in all_outputs])
            means[k] = float(np.mean(vals))
            stds[k] = float(np.std(vals))
        return means, stds

    def score_with_uq(self, seq: str = None, candidate=None, n_passes: int = 50) -> Dict:
        results = {}
        window = 50
        if seq is not None:
            chunk = seq[:window]
            tokens = self.encode_seq(chunk).to(self.device)
            if self.aav_model:
                aav_means, aav_stds = self._mc_predict(
                    self.aav_model, tokens,
                    lambda o: {
                        "cardiac_score": float(o["tissue_scores"][0, 0]),
                        "immune_score": float(o["immune_score"][0, 0]),
                        "delivery_score": float(o["delivery_score"][0, 0]),
                    },
                    n_passes,
                )
                results["aav"] = {
                    "mean": aav_means,
                    "std": aav_stds,
                    "ci95_lower": {k: float(np.clip(aav_means[k] - 1.96 * aav_stds[k], 0, 1)) for k in aav_means},
                    "ci95_upper": {k: float(np.clip(aav_means[k] + 1.96 * aav_stds[k], 0, 1)) for k in aav_means},
                }
            if self.immune_model:
                imm_means, imm_stds = self._mc_predict(
                    self.immune_model, tokens,
                    lambda o: {
                        "total_escape": float(o["total_escape"][0]),
                        "resistance": float(o["resistance"][0]),
                    },
                    n_passes,
                )
                results["immune"] = {
                    "mean": imm_means,
                    "std": imm_stds,
                    "ci95_lower": {k: float(np.clip(imm_means[k] - 1.96 * imm_stds[k], 0, 1)) for k in imm_means},
                    "ci95_upper": {k: float(np.clip(imm_means[k] + 1.96 * imm_stds[k], 0, 1)) for k in imm_means},
                }
        if candidate is not None and self.lnp_model:
            lipid_map = {"DLin-MC3-DMA": 0, "SM-102": 1, "ALC-0315": 2,
                         "DODAP": 3, "DLin-DMA": 4, "cKK-E11": 5}
            peg_map = {"DMG-PEG2000": 0, "DSPC-PEG2000": 1, "DSPE-PEG2000": 2}
            helper_map = {"DSPC": 0, "DPPC": 1, "DOPE": 2, "POPC": 3}
            feats = torch.tensor([[
                lipid_map.get(getattr(candidate, "ionizable_lipid", "DLin-MC3-DMA"), 0),
                peg_map.get(getattr(candidate, "peg_lipid", "DMG-PEG2000"), 0),
                helper_map.get(getattr(candidate, "helper_lipid", "DSPC"), 0),
                getattr(candidate, "ionizable_frac", 0.40),
                getattr(candidate, "peg_frac", 0.015),
                getattr(candidate, "cholesterol_frac", 0.35),
                getattr(candidate, "pka", 6.3),
                16 / 22.0,
                2 / 5.0,
            ]], dtype=torch.float32).to(self.device)
            lnp_means, lnp_stds = self._mc_predict(
                self.lnp_model, feats,
                lambda o: {
                    "cardiac_delivery": float(o["cardiac_delivery"][0]),
                    "hepatic_avoidance": float(o["hepatic_avoidance"][0]),
                },
                n_passes,
            )
            results["lnp"] = {
                "mean": lnp_means,
                "std": lnp_stds,
                "ci95_lower": {k: float(np.clip(lnp_means[k] - 1.96 * lnp_stds[k], 0, 1)) for k in lnp_means},
                "ci95_upper": {k: float(np.clip(lnp_means[k] + 1.96 * lnp_stds[k], 0, 1)) for k in lnp_means},
            }
        return results

    def score_aav(self, seq: str) -> dict:
        if self.aav_model is None:
            return {"cardiac_score": 0.5, "immune_score": 0.5, "delivery_score": 0.5}

        window = 50
        with torch.no_grad():
            chunk = seq[:window]
            tokens = self.encode_seq(chunk).to(self.device)
            out = self.aav_model(tokens)
            cardiac_score = float(out["tissue_scores"][0, 0])
            immune_score = float(out["immune_score"][0, 0])
            delivery_score = float(out["delivery_score"][0, 0])

        return {
            "cardiac_score": cardiac_score,
            "immune_score": immune_score,
            "delivery_score": delivery_score,
        }

    def score_lnp(self, candidate) -> dict:
        if self.lnp_model is None or not hasattr(candidate, "ionizable_lipid"):
            return {"cardiac_delivery": 0.5, "hepatic_avoidance": 0.5}

        lipid_map = {"DLin-MC3-DMA": 0, "SM-102": 1, "ALC-0315": 2,
                     "DODAP": 3, "DLin-DMA": 4, "cKK-E11": 5}
        peg_map = {"DMG-PEG2000": 0, "DSPC-PEG2000": 1, "DSPE-PEG2000": 2}
        helper_map = {"DSPC": 0, "DPPC": 1, "DOPE": 2, "POPC": 3}

        feats = torch.tensor([[
            lipid_map.get(getattr(candidate, "ionizable_lipid", "DLin-MC3-DMA"), 0),
            peg_map.get(getattr(candidate, "peg_lipid", "DMG-PEG2000"), 0),
            helper_map.get(getattr(candidate, "helper_lipid", "DSPC"), 0),
            getattr(candidate, "ionizable_frac", 0.40),
            getattr(candidate, "peg_frac", 0.015),
            getattr(candidate, "cholesterol_frac", 0.35),
            getattr(candidate, "pka", 6.3),
            16 / 22.0,
            2 / 5.0,
        ]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.lnp_model(feats)

        return {
            "cardiac_delivery": float(out["cardiac_delivery"][0]),
            "hepatic_avoidance": float(out["hepatic_avoidance"][0]),
        }

    def score_immune(self, seq: str) -> dict:
        if self.immune_model is None:
            return {"total_escape": 0.5, "resistance": 0.5}

        window = 50
        with torch.no_grad():
            chunk = seq[:window]
            tokens = self.encode_seq(chunk).to(self.device)
            out = self.immune_model(tokens)

        return {
            "total_escape": float(out["total_escape"][0]),
            "resistance": float(out["resistance"][0]),
        }
