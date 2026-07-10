import os
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


class AAVSequenceDataset(Dataset):
    def __init__(self, data_path: str, max_len: int = 750):
        import json
        with open(data_path) as f:
            self.data = json.load(f)
        self.max_len = max_len
        self.tissues = ["cardiac", "neuronal", "joint_cartilage", "skeletal_muscle",
                        "hepatic", "renal", "pulmonary", "adipose"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item.get("sequence", "A" * self.max_len)[:self.max_len]
        encoded = torch.zeros(self.max_len, dtype=torch.long)
        for i, aa in enumerate(seq):
            if aa in AA_TO_IDX:
                encoded[i] = AA_TO_IDX[aa]

        tropism = item.get("tropism_target", item.get("tissue_target", "cardiac"))
        tissue_target = torch.tensor(
            self.tissues.index(tropism) if tropism in self.tissues else 0,
            dtype=torch.long
        )
        tissue_scores = torch.tensor(
            [item.get("tissue_scores", {}).get(t, 0.5) for t in self.tissues],
            dtype=torch.float32
        )
        delivery_eff = torch.tensor(item.get("delivery_efficiency", 0.5), dtype=torch.float32)
        immune_escape = torch.tensor(item.get("immune_escape_score", item.get("total_escape_score", 0.5)), dtype=torch.float32)

        return {
            "sequence": encoded,
            "tissue_target": tissue_target,
            "tissue_scores": tissue_scores,
            "delivery_efficiency": delivery_eff,
            "immune_escape": immune_escape,
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AAVTropismTransformer(nn.Module):
    def __init__(self, vocab_size=20, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, num_tissues=8, max_seq_len=750):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.tissue_classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_tissues),
        )

        self.delivery_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.immune_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.tissue_score_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_tissues),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x)

        pooled = encoded.mean(dim=1)

        tissue_logits = self.tissue_classifier(pooled)
        delivery_score = self.delivery_head(pooled)
        immune_score = self.immune_head(pooled)
        tissue_scores = self.tissue_score_head(pooled)

        return {
            "tissue_logits": tissue_logits,
            "delivery_score": delivery_score,
            "immune_score": immune_score,
            "tissue_scores": tissue_scores,
            "encoded": pooled,
        }

    def _generate_square_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.float().masked_fill(mask == 1, float("-inf"))
        return mask


class AAVTropismLoss(nn.Module):
    def __init__(self, tissue_weight=1.0, delivery_weight=1.0, immune_weight=0.5):
        super().__init__()
        self.tissue_weight = tissue_weight
        self.delivery_weight = delivery_weight
        self.immune_weight = immune_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        tissue_loss = self.ce_loss(predictions["tissue_logits"], targets["tissue_target"])
        delivery_loss = self.mse_loss(
            predictions["delivery_score"].squeeze(), targets["delivery_efficiency"]
        )
        immune_score = predictions["immune_score"].squeeze().clamp(0.001, 0.999)
        immune_loss = self.bce_loss(immune_score, targets["immune_escape"])
        tissue_score_loss = self.mse_loss(predictions["tissue_scores"], targets["tissue_scores"])

        total = (self.tissue_weight * tissue_loss +
                 self.delivery_weight * delivery_loss +
                 self.immune_weight * immune_loss +
                 0.3 * tissue_score_loss)

        return {
            "total": total,
            "tissue": tissue_loss,
            "delivery": delivery_loss,
            "immune": immune_loss,
            "tissue_scores": tissue_score_loss,
        }


class LNPDeliveryDataset(Dataset):
    def __init__(self, data_path: str):
        import json
        with open(data_path) as f:
            self.data = json.load(f)
        self.lipid_to_idx = {
            "DLin-MC3-DMA": 0, "SM-102": 1, "ALC-0315": 2,
            "DODAP": 3, "DLin-DMA": 4, "cKK-E11": 5,
        }
        self.peg_to_idx = {"DMG-PEG2000": 0, "DSPC-PEG2000": 1, "DSPE-PEG2000": 2}
        self.helper_to_idx = {"DSPC": 0, "DPPC": 1, "DOPE": 2, "POPC": 3}
        self.organs = ["heart_organoid", "brain_organoid", "liver_organoid", "joint_organoid"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor([
            self.lipid_to_idx.get(item.get("ionizable_lipid", "DLin-MC3-DMA"), 0),
            self.peg_to_idx.get(item.get("peg_lipid", "DMG-PEG2000"), 0),
            self.helper_to_idx.get(item.get("helper_lipid", "DSPC"), 0),
            item.get("ionizable_frac", 0.40),
            item.get("peg_frac", 0.015),
            item.get("cholesterol_frac", 0.35),
            item.get("pka", 6.3),
            item.get("tail_length", 16) / 22.0,
            item.get("unsaturation", 2) / 5.0,
        ], dtype=torch.float32)

        organoid_counts = torch.tensor(
            [item.get("organoid_barcode_counts", {}).get(o, 500) for o in self.organs],
            dtype=torch.float32
        ) / 1000.0

        return {
            "features": features,
            "delivery_efficiency": torch.tensor(item.get("delivery_efficiency", 0.5), dtype=torch.float32),
            "organoid_counts": organoid_counts,
            "immune_activation": torch.tensor(item.get("immune_activation", 0.2), dtype=torch.float32),
            "cytotoxicity": torch.tensor(item.get("cytotoxicity", 0.1), dtype=torch.float32),
        }


class LNPDeliveryMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, output_dim=1, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
        )
        self.delivery_head = nn.Linear(hidden_dim // 4, 1)
        self.organoid_head = nn.Linear(hidden_dim // 4, 4)
        self.safety_head = nn.Linear(hidden_dim // 4, 2)

    def forward(self, x):
        h = self.network(x)
        return {
            "delivery_score": torch.sigmoid(self.delivery_head(h)).squeeze(-1),
            "organoid_counts": torch.relu(self.organoid_head(h)),
            "safety": torch.sigmoid(self.safety_head(h)),
        }


class LNPDeliveryLoss(nn.Module):
    def __init__(self, delivery_weight=1.0, organoid_weight=0.5, safety_weight=0.3):
        super().__init__()
        self.delivery_weight = delivery_weight
        self.organoid_weight = organoid_weight
        self.safety_weight = safety_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        delivery_loss = self.mse_loss(predictions["delivery_score"], targets["delivery_efficiency"])
        organoid_loss = self.mse_loss(predictions["organoid_counts"], targets["organoid_counts"])
        immune_loss = self.bce_loss(predictions["safety"][:, 0], targets["immune_activation"])
        cytotoxicity_loss = self.bce_loss(predictions["safety"][:, 1], 1.0 - targets["cytotoxicity"])

        total = (self.delivery_weight * delivery_loss +
                 self.organoid_weight * organoid_loss +
                 self.safety_weight * (immune_loss + cytotoxicity_loss))

        return {
            "total": total,
            "delivery": delivery_loss,
            "organoid": organoid_loss,
            "immune": immune_loss,
            "cytotoxicity": cytotoxicity_loss,
        }


class ImmuneEscapeDataset(Dataset):
    def __init__(self, data_path: str, max_len: int = 750):
        import json
        with open(data_path) as f:
            self.data = json.load(f)
        self.max_len = max_len
        self.antibodies = ["AAV2_Ab4", "AAV2_Ab58", "AAV8_Ab1", "AAV9_Ab3",
                           "human_IgG_pool", "human_IgM_pool", "anti-AAV9_serum"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item.get("sequence", "A" * self.max_len)[:self.max_len]
        encoded = torch.zeros(self.max_len, dtype=torch.long)
        for i, aa in enumerate(seq):
            if aa in AA_TO_IDX:
                encoded[i] = AA_TO_IDX[aa]

        ab_responses = item.get("antibody_responses", {})
        escape_scores = torch.tensor(
            [ab_responses.get(ab, {}).get("escape_score", 0.5) for ab in self.antibodies],
            dtype=torch.float32
        )
        binding_energies = torch.tensor(
            [ab_responses.get(ab, {}).get("binding_energy", -5.0) for ab in self.antibodies],
            dtype=torch.float32
        )

        return {
            "sequence": encoded,
            "escape_scores": escape_scores,
            "binding_energies": binding_energies,
            "total_escape": torch.tensor(item.get("total_escape_score", 0.5), dtype=torch.float32),
            "neutralization_resistance": torch.tensor(item.get("neutralization_resistance", 0.5), dtype=torch.float32),
        }


class ImmuneEscapeTransformer(nn.Module):
    def __init__(self, vocab_size=20, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, num_antibodies=7, max_seq_len=750):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.escape_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_antibodies), nn.Sigmoid(),
        )
        self.binding_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_antibodies),
        )
        self.total_escape_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.resistance_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)
        return {
            "escape_scores": self.escape_head(pooled),
            "binding_energies": self.binding_head(pooled),
            "total_escape": self.total_escape_head(pooled).squeeze(-1),
            "resistance": self.resistance_head(pooled).squeeze(-1),
        }


class ImmuneEscapeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        escape_loss = self.bce(pred["escape_scores"].clamp(0.001, 0.999), target["escape_scores"])
        binding_loss = self.mse(pred["binding_energies"], target["binding_energies"])
        total_loss = self.bce(pred["total_escape"].clamp(0.001, 0.999), target["total_escape"])
        resistance_loss = self.bce(pred["resistance"].clamp(0.001, 0.999), target["neutralization_resistance"])
        total = escape_loss + 0.5 * binding_loss + total_loss + resistance_loss
        return {"total": total, "escape": escape_loss, "binding": binding_loss,
                "total_escape": total_loss, "resistance": resistance_loss}
