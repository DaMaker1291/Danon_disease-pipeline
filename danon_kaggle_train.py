"""
DANON DISEASE - MODEL FINE-TUNING ON KAGGLE
============================================
Fine-tunes the 3 existing models for Danon Disease (AAV9-LAMP2B):
  - AAV Tropism Transformer -> Cardiac myocyte targeting
  - LNP Delivery MLP -> Cardiac delivery, hepatic avoidance
  - Immune Escape Transformer -> AAV9-specific immune evasion

SETUP:
  1. Upload data/colab_subsampled/ as Kaggle dataset
  2. Open Kaggle -> Code -> New Notebook
  3. Attach dataset
  4. Set GPU accelerator (P100 or T4)
  5. Run All
"""

import os
import glob
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
USE_CUDA = False
if torch.cuda.is_available():
    try:
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
        if cap[0] >= 7:
            USE_CUDA = True
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
            print(f"VRAM: {vram / 1e9:.1f} GB")
    except Exception as e:
        print(f"CUDA test failed ({e}), using CPU")
else:
    print("No CUDA available, using CPU")
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Device: {device}")

# ========================= DATA =========================
DATA_DIR = "/kaggle/input/longevity-real-data"
for p in glob.glob("/kaggle/input/**/real_screening_aav_tropism.json", recursive=True):
    DATA_DIR = os.path.dirname(p)
    break
else:
    for p in glob.glob("/kaggle/input/*/*"):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "real_screening_aav_tropism.json")):
            DATA_DIR = p
            break

import glob
print(f"Data: {DATA_DIR}")

def safe_load(path):
    with open(path) as f:
        data = json.load(f)
    cleaned = [d for d in data if all(
        isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v))
        for v in d.values() if isinstance(v, (int, float))
    )]
    return cleaned

aav_data = safe_load(f"{DATA_DIR}/real_screening_aav_tropism.json")
lnp_data = safe_load(f"{DATA_DIR}/real_screening_lnp_delivery.json")
immune_data = safe_load(f"{DATA_DIR}/real_screening_immune_escape.json")

print(f"AAV: {len(aav_data)} | LNP: {len(lnp_data)} | Immune: {len(immune_data)}")

# ========================= REWEIGHT FOR DANON =========================
# Boost cardiac tissue samples to 60% of AAV data for Danon focus
cardiac_samples = [d for d in aav_data if d.get("tropism_target") == "cardiac"]
non_cardiac = [d for d in aav_data if d.get("tropism_target") != "cardiac"]
target_cardiac = int(len(aav_data) * 0.60)
if len(cardiac_samples) < target_cardiac and non_cardiac:
    oversample = np.random.choice(len(cardiac_samples), size=target_cardiac - len(cardiac_samples), replace=True)
    cardiac_augmented = [cardiac_samples[i] for i in oversample]
    aav_danon = cardiac_samples + cardiac_augmented + non_cardiac
else:
    aav_danon = aav_data
np.random.shuffle(aav_danon)
print(f"Danon AAV reweighted: {len(aav_danon)} ({len(cardiac_samples)} cardiac originals)")

# ========================= ENCODING =========================
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

# ========================= DATASETS =========================
class AAVDanonDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
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

        tropism = item.get("tropism_target", "cardiac")
        tissue_target = torch.tensor(
            self.tissues.index(tropism) if tropism in self.tissues else 0,
            dtype=torch.long
        )
        tissue_scores = torch.tensor(
            [item.get("tissue_scores", {}).get(t, 0.5) for t in self.tissues],
            dtype=torch.float32
        )
        delivery_eff = torch.tensor(item.get("delivery_efficiency", 0.5), dtype=torch.float32)
        immune_escape = torch.tensor(
            item.get("immune_escape_score", item.get("total_escape_score", 0.5)),
            dtype=torch.float32
        )
        cardiac_affinity = torch.tensor(
            item.get("tissue_scores", {}).get("cardiac", 0.5), dtype=torch.float32
        )
        hepatic_penalty = torch.tensor(
            1.0 - item.get("tissue_scores", {}).get("hepatic", 0.5), dtype=torch.float32
        )
        return {
            "sequence": encoded,
            "tissue_target": tissue_target,
            "tissue_scores": tissue_scores,
            "delivery_efficiency": delivery_eff,
            "immune_escape": immune_escape,
            "cardiac_affinity": cardiac_affinity,
            "hepatic_penalty": hepatic_penalty,
        }


class LNPDanonDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.lipid_to_idx = {
            "DLin-MC3-DMA": 0, "SM-102": 1, "ALC-0315": 2,
            "DODAP": 3, "DLin-DMA": 4, "cKK-E11": 5,
        }
        self.peg_to_idx = {"DMG-PEG2000": 0, "DSPC-PEG2000": 1, "DSPE-PEG2000": 2}
        self.helper_to_idx = {"DSPC": 0, "DPPC": 1, "DOPE": 2, "POPC": 3}

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

        cardiac_delivery = torch.tensor(
            item.get("organoid_barcode_counts", {}).get("heart_organoid", 500) / 1000.0,
            dtype=torch.float32
        )
        hepatic_avoidance = torch.tensor(
            1.0 - item.get("organoid_barcode_counts", {}).get("liver_organoid", 500) / 1000.0,
            dtype=torch.float32
        )
        return {
            "features": features,
            "delivery_efficiency": torch.tensor(item.get("delivery_efficiency", 0.5), dtype=torch.float32),
            "cardiac_delivery": cardiac_delivery,
            "hepatic_avoidance": hepatic_avoidance,
            "immune_activation": torch.tensor(item.get("immune_activation", 0.2), dtype=torch.float32),
            "cytotoxicity": torch.tensor(item.get("cytotoxicity", 0.1), dtype=torch.float32),
        }


class ImmuneDanonDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
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

# ========================= MODELS (same architecture, Danon heads) =========================
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
        self.tissue_classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_tissues),
        )
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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled = encoded.mean(dim=1)
        return {
            "tissue_logits": self.tissue_classifier(pooled),
            "delivery_score": self.delivery_head(pooled),
            "immune_score": self.immune_head(pooled),
            "tissue_scores": self.tissue_score_head(pooled),
            "encoded": pooled,
        }


class AAVDanonLoss(nn.Module):
    """Danon-specific loss: 60% cardiac, 20% hepatic avoidance, 20% delivery."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, targets):
        tissue_loss = self.ce(pred["tissue_logits"], targets["tissue_target"])
        delivery_loss = self.mse(pred["delivery_score"].squeeze(), targets["delivery_efficiency"])
        immune_loss = self.bce(pred["immune_score"].squeeze().clamp(0.001, 0.999), targets["immune_escape"])
        tissue_score_loss = self.mse(pred["tissue_scores"], targets["tissue_scores"])

        cardiac_idx = 0
        cardiac_pred = pred["tissue_scores"][:, cardiac_idx]
        cardiac_loss = self.mse(cardiac_pred, targets.get("cardiac_affinity", torch.ones_like(cardiac_pred) * 0.7))

        hepatic_idx = 4
        hepatic_pred = pred["tissue_scores"][:, hepatic_idx]
        hepatic_loss = self.mse(hepatic_pred, torch.zeros_like(hepatic_pred))

        total = (0.30 * tissue_loss + 0.15 * delivery_loss + 0.10 * immune_loss +
                 0.15 * tissue_score_loss + 0.20 * cardiac_loss + 0.10 * hepatic_loss)
        return {"total": total, "tissue": tissue_loss, "delivery": delivery_loss,
                "immune": immune_loss, "cardiac": cardiac_loss, "hepatic": hepatic_loss}


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
        self.safety_head = nn.Linear(hidden_dim // 4, 2)

    def forward(self, x):
        h = self.network(x)
        return {
            "delivery_score": torch.sigmoid(self.delivery_head(h)).squeeze(-1),
            "cardiac_delivery": torch.sigmoid(self.cardiac_head(h)).squeeze(-1),
            "hepatic_avoidance": torch.sigmoid(self.hepatic_head(h)).squeeze(-1),
            "safety": torch.sigmoid(self.safety_head(h)),
        }


class LNPDanonLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, targets):
        delivery_loss = self.mse(pred["delivery_score"], targets["delivery_efficiency"])
        cardiac_loss = self.mse(pred["cardiac_delivery"], targets["cardiac_delivery"])
        hepatic_loss = self.mse(pred["hepatic_avoidance"], targets["hepatic_avoidance"])
        immune_loss = self.bce(pred["safety"][:, 0], targets["immune_activation"])
        cytotoxicity_loss = self.bce(pred["safety"][:, 1], 1.0 - targets["cytotoxicity"])
        total = (0.25 * delivery_loss + 0.30 * cardiac_loss + 0.20 * hepatic_loss +
                 0.15 * immune_loss + 0.10 * cytotoxicity_loss)
        return {"total": total, "delivery": delivery_loss, "cardiac": cardiac_loss,
                "hepatic": hepatic_loss, "immune": immune_loss, "cytotoxicity": cytotoxicity_loss}


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
        self.escape_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_antibodies), nn.Sigmoid(),
        )
        self.binding_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_antibodies),
        )
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


# ========================= TRAINING =========================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


CKPT = "/kaggle/working/checkpoints_danon"
os.makedirs(CKPT, exist_ok=True)

# --- AAV ---
print("\n" + "=" * 60)
print("DANON FINE-TUNE: AAV Tropism Transformer")
print("=" * 60)

aav_ds = AAVDanonDataset(aav_danon, max_len=50)
val_size = int(len(aav_ds) * 0.1)
train_ds, val_ds = random_split(aav_ds, [len(aav_ds) - val_size, val_size])
aav_loader = DataLoader(train_ds, batch_size=256 if USE_CUDA else 32, shuffle=True, num_workers=2, pin_memory=USE_CUDA)
aav_val_loader = DataLoader(val_ds, batch_size=256 if USE_CUDA else 32, shuffle=False, num_workers=2)

aav_model = AAVTropismTransformer().to(device)
aav_opt = AdamW(aav_model.parameters(), lr=3e-4, weight_decay=1e-4)
aav_sched = CosineAnnealingWarmRestarts(aav_opt, T_0=10, T_mult=2)
aav_criterion = AAVDanonLoss()
aav_stop = EarlyStopping(patience=10)
aav_scaler = torch.amp.GradScaler(device.type, enabled=USE_CUDA)

aav_hist = {"train_loss": [], "val_loss": []}
best_val = float("inf")
aav_epochs = 50 if USE_CUDA else 10

for epoch in range(aav_epochs):
    t0 = time.time()
    aav_model.train()
    train_loss = 0
    n = 0
    for batch in aav_loader:
        seqs = batch["sequence"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
        with torch.amp.autocast(device_type=device.type, enabled=USE_CUDA):
            pred = aav_model(seqs)
            loss = aav_criterion(pred, targets)["total"]
        aav_scaler.scale(loss).backward()
        aav_scaler.unscale_(aav_opt)
        nn.utils.clip_grad_norm_(aav_model.parameters(), 1.0)
        aav_scaler.step(aav_opt)
        aav_scaler.update()
        aav_opt.zero_grad()
        train_loss += loss.item()
        n += 1
    train_loss /= n
    aav_sched.step()

    aav_model.eval()
    val_loss = 0
    vn = 0
    with torch.no_grad():
        for batch in aav_val_loader:
            seqs = batch["sequence"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
            pred = aav_model(seqs)
            val_loss += aav_criterion(pred, targets)["total"].item()
            vn += 1
    val_loss /= max(vn, 1)
    aav_hist["train_loss"].append(train_loss)
    aav_hist["val_loss"].append(val_loss)

    marker = ""
    if val_loss < best_val:
        best_val = val_loss
        marker = " *"
        torch.save(aav_model.state_dict(), f"{CKPT}/aav_danon_best.pt")

    if (epoch + 1) % 5 == 0 or marker:
        print(f"  Epoch {epoch+1:3d}/{aav_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {time.time()-t0:.1f}s{marker}")

    aav_stop.step(val_loss)
    if aav_stop.should_stop:
        break

torch.save(aav_model.state_dict(), f"{CKPT}/aav_danon_final.pt")
print(f"  Best: {best_val:.4f}")

# --- LNP ---
print("\n" + "=" * 60)
print("DANON FINE-TUNE: LNP Delivery MLP")
print("=" * 60)

lnp_ds = LNPDanonDataset(lnp_data)
val_size = int(len(lnp_ds) * 0.15)
train_ds, val_ds = random_split(lnp_ds, [len(lnp_ds) - val_size, val_size])
lnp_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
lnp_val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

lnp_model = LNPDeliveryMLP().to(device)
lnp_opt = AdamW(lnp_model.parameters(), lr=5e-4, weight_decay=1e-4)
lnp_sched = CosineAnnealingWarmRestarts(lnp_opt, T_0=10, T_mult=2)
lnp_criterion = LNPDanonLoss()
lnp_stop = EarlyStopping(patience=20)

lnp_hist = {"train_loss": [], "val_loss": []}
best_val = float("inf")
lnp_epochs = 100 if USE_CUDA else 20

for epoch in range(lnp_epochs):
    t0 = time.time()
    lnp_model.train()
    train_loss = 0
    n = 0
    for batch in lnp_loader:
        feats = batch["features"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "features"}
        pred = lnp_model(feats)
        loss = lnp_criterion(pred, targets)["total"]
        lnp_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(lnp_model.parameters(), 1.0)
        lnp_opt.step()
        train_loss += loss.item()
        n += 1
    train_loss /= n
    lnp_sched.step()

    lnp_model.eval()
    val_loss = 0
    vn = 0
    with torch.no_grad():
        for batch in lnp_val_loader:
            feats = batch["features"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "features"}
            pred = lnp_model(feats)
            val_loss += lnp_criterion(pred, targets)["total"].item()
            vn += 1
    val_loss /= max(vn, 1)
    lnp_hist["train_loss"].append(train_loss)
    lnp_hist["val_loss"].append(val_loss)

    marker = ""
    if val_loss < best_val:
        best_val = val_loss
        marker = " *"
        torch.save(lnp_model.state_dict(), f"{CKPT}/lnp_danon_best.pt")

    if (epoch + 1) % 10 == 0 or marker:
        print(f"  Epoch {epoch+1:3d}/{lnp_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {time.time()-t0:.1f}s{marker}")

    lnp_stop.step(val_loss)
    if lnp_stop.should_stop:
        break

torch.save(lnp_model.state_dict(), f"{CKPT}/lnp_danon_final.pt")
print(f"  Best: {best_val:.4f}")

# --- Immune ---
print("\n" + "=" * 60)
print("DANON FINE-TUNE: Immune Escape Transformer")
print("=" * 60)

imm_ds = ImmuneDanonDataset(immune_data, max_len=50)
val_size = int(len(imm_ds) * 0.1)
train_ds, val_ds = random_split(imm_ds, [len(imm_ds) - val_size, val_size])
imm_loader = DataLoader(train_ds, batch_size=256 if USE_CUDA else 32, shuffle=True, num_workers=2, pin_memory=USE_CUDA)
imm_val_loader = DataLoader(val_ds, batch_size=256 if USE_CUDA else 32, shuffle=False, num_workers=2)

imm_model = ImmuneEscapeTransformer().to(device)
imm_opt = AdamW(imm_model.parameters(), lr=3e-4, weight_decay=1e-4)
imm_sched = CosineAnnealingWarmRestarts(imm_opt, T_0=10, T_mult=2)
imm_criterion = ImmuneEscapeLoss()
imm_stop = EarlyStopping(patience=10)
imm_scaler = torch.amp.GradScaler(device.type, enabled=USE_CUDA)

imm_hist = {"train_loss": [], "val_loss": []}
best_val = float("inf")
imm_epochs = 50 if USE_CUDA else 10

for epoch in range(imm_epochs):
    t0 = time.time()
    imm_model.train()
    train_loss = 0
    n = 0
    for batch in imm_loader:
        seqs = batch["sequence"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
        with torch.amp.autocast(device_type=device.type, enabled=USE_CUDA):
            pred = imm_model(seqs)
            loss = imm_criterion(pred, targets)["total"]
        imm_scaler.scale(loss).backward()
        imm_scaler.unscale_(imm_opt)
        nn.utils.clip_grad_norm_(imm_model.parameters(), 1.0)
        imm_scaler.step(imm_opt)
        imm_scaler.update()
        imm_opt.zero_grad()
        train_loss += loss.item()
        n += 1
    train_loss /= n
    imm_sched.step()

    imm_model.eval()
    val_loss = 0
    vn = 0
    with torch.no_grad():
        for batch in imm_val_loader:
            seqs = batch["sequence"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
            pred = imm_model(seqs)
            val_loss += imm_criterion(pred, targets)["total"].item()
            vn += 1
    val_loss /= max(vn, 1)
    imm_hist["train_loss"].append(train_loss)
    imm_hist["val_loss"].append(val_loss)

    marker = ""
    if val_loss < best_val:
        best_val = val_loss
        marker = " *"
        torch.save(imm_model.state_dict(), f"{CKPT}/immune_danon_best.pt")

    if (epoch + 1) % 5 == 0 or marker:
        print(f"  Epoch {epoch+1:3d}/{imm_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {time.time()-t0:.1f}s{marker}")

    imm_stop.step(val_loss)
    if imm_stop.should_stop:
        break

torch.save(imm_model.state_dict(), f"{CKPT}/immune_danon_final.pt")
print(f"  Best: {best_val:.4f}")

# ========================= PLOTS =========================
def plot_danon(history, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train", color="crimson", lw=2)
    ax.plot(history["val_loss"], label="Val", color="dodgerblue", lw=2, ls="--")
    ax.set_title(f"Danon Disease - {title}", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")

plot_danon(aav_hist, "AAV Tropism", "/kaggle/working/danon_aav_loss.png")
plot_danon(lnp_hist, "LNP Delivery", "/kaggle/working/danon_lnp_loss.png")
plot_danon(imm_hist, "Immune Escape", "/kaggle/working/danon_immune_loss.png")

# ========================= SUMMARY =========================
print("\n" + "=" * 60)
print("DANON DISEASE FINE-TUNING COMPLETE")
print("=" * 60)
for pt in Path(CKPT).glob("*.pt"):
    print(f"  {pt.name}: {pt.stat().st_size/1024/1024:.1f} MB")

for name, hist in [("aav", aav_hist), ("lnp", lnp_hist), ("immune", imm_hist)]:
    with open(f"/kaggle/working/danon_{name}_history.json", "w") as f:
        json.dump(hist, f)

print("\nDownload: Output -> Download")
print("Models ready for danon_main.py inference")
