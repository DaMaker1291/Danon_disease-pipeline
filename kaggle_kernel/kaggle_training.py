"""
=============================================================================
LONGEVITY PIPELINE - KAGGLE TRAINING NOTEBOOK
=============================================================================
Train on FULL real data (100K AAV + 773 LNP) with P100 GPU.

SETUP:
  1. Create Kaggle dataset with export_for_kaggle.py
  2. Open Kaggle -> Notebooks -> New Notebook
  3. Attach dataset: supro/longevity-real-data
  4. Set accelerator: GPU (P100 or T4)
  5. Copy this script into a cell, or upload as notebook
  6. Run All

Trains 3 models:
  - AAV Tropism Transformer (100K 7-AA capsid variants)
  - LNP Delivery MLP (773 LNP formulations)
  - Immune Escape Transformer (100K variants)
=============================================================================
"""

# =============================================================================
# CELL 1: Imports & Setup
# =============================================================================
import os
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

# Try CUDA, fall back to CPU if incompatible GPU
USE_CUDA = False
if torch.cuda.is_available():
    try:
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
        if cap[0] < 7:
            print(f"GPU compute capability {cap[0]}.{cap[1]} < 7.0, incompatible with PyTorch {torch.__version__}")
            print("Falling back to CPU")
        else:
            test = torch.zeros(1).cuda() + torch.zeros(1).cuda()
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

# =============================================================================
# CELL 2: Load Data
# =============================================================================
# Kaggle dataset path (change if using different dataset name)
DATA_DIR = "/kaggle/input/longevity-real-data"

# If running locally, override:
# DATA_DIR = "./data"

# Auto-discover dataset path
import glob, os
candidates = glob.glob("/kaggle/input/**/real_screening_aav_tropism.json", recursive=True)
if candidates:
    DATA_DIR = os.path.dirname(candidates[0])
    print(f"Auto-discovered dataset at: {DATA_DIR}")
else:
    # Try common paths
    for p in glob.glob("/kaggle/input/*/*"):
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "real_screening_aav_tropism.json")):
                DATA_DIR = p
                break
    else:
        DATA_DIR = "/kaggle/input/longevity-real-data"
    print(f"Using path: {DATA_DIR}")
    print(f"Contents of /kaggle/input/:")
    for root, dirs, files in os.walk("/kaggle/input/"):
        depth = root.replace("/kaggle/input/", "").count(os.sep)
        if depth <= 2:
            print(f"  {'  ' * depth}{os.path.basename(root)}/")
            if depth <= 1:
                for f in files[:5]:
                    print(f"  {'  ' * (depth+1)}{f}")

aav_path = f"{DATA_DIR}/real_screening_aav_tropism.json"
lnp_path = f"{DATA_DIR}/real_screening_lnp_delivery.json"
immune_path = f"{DATA_DIR}/real_screening_immune_escape.json"

print(f"\nLoading data from {DATA_DIR}...")
t0 = time.time()
with open(aav_path) as f:
    aav_data = json.load(f)
print(f"  AAV: {len(aav_data)} samples ({time.time()-t0:.1f}s)")

t0 = time.time()
with open(lnp_path) as f:
    lnp_data = json.load(f)
print(f"  LNP: {len(lnp_data)} samples ({time.time()-t0:.1f}s)")

t0 = time.time()
with open(immune_path) as f:
    immune_data = json.load(f)
print(f"  Immune: {len(immune_data)} samples ({time.time()-t0:.1f}s)")

def clean_nan(data, name):
    def has_nan_recursive(obj):
        if isinstance(obj, float):
            return math.isnan(obj) or math.isinf(obj)
        if isinstance(obj, dict):
            return any(has_nan_recursive(v) for v in obj.values())
        if isinstance(obj, list):
            return any(has_nan_recursive(v) for v in obj)
        return False
    cleaned = [item for item in data if not has_nan_recursive(item)]
    print(f"  {name}: kept {len(cleaned)}, dropped {len(data) - len(cleaned)} NaN/Inf samples")
    return cleaned

aav_data = clean_nan(aav_data, "AAV")
lnp_data = clean_nan(lnp_data, "LNP")
immune_data = clean_nan(immune_data, "Immune")

# =============================================================================
# CELL 3: Amino Acid Encoding
# =============================================================================
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

# =============================================================================
# CELL 4: Datasets (max_len=50 for 7-AA sequences)
# =============================================================================
class AAVSequenceDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
        self.max_len = max_len
        self.tissues = ["cardiac", "neuronal", "joint_cartilage", "skeletal_muscle",
                        "hepatic", "renal", "pulmonary", "adipose"]

    @staticmethod
    def safe_float(val, default=0.5):
        if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
            return float(val)
        return default

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
        def safe_float(val, default=0.5):
            if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
                return float(val)
            return default

        tissue_scores = torch.tensor(
            [self.safe_float(item.get("tissue_scores", {}).get(t, 0.5)) for t in self.tissues],
            dtype=torch.float32
        )
        delivery_eff = torch.tensor(self.safe_float(item.get("delivery_efficiency", 0.5)), dtype=torch.float32)
        immune_escape = torch.tensor(self.safe_float(item.get("immune_escape_score", item.get("total_escape_score", 0.5))), dtype=torch.float32)
        return {
            "sequence": encoded,
            "tissue_target": tissue_target,
            "tissue_scores": tissue_scores,
            "delivery_efficiency": delivery_eff,
            "immune_escape": immune_escape,
        }


class LNPDeliveryDataset(Dataset):
    def __init__(self, data):
        self.data = data
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


class ImmuneEscapeDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
        self.max_len = max_len
        self.antibodies = ["AAV2_Ab4", "AAV2_Ab58", "AAV8_Ab1", "AAV9_Ab3",
                           "human_IgG_pool", "human_IgM_pool", "anti-AAV9_serum"]

    @staticmethod
    def safe_float(val, default=0.5):
        if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
            return float(val)
        return default

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
            [self.safe_float(ab_responses.get(ab, {}).get("escape_score", 0.5)) for ab in self.antibodies],
            dtype=torch.float32
        )
        binding_energies = torch.tensor(
            [self.safe_float(ab_responses.get(ab, {}).get("binding_energy", -5.0)) for ab in self.antibodies],
            dtype=torch.float32
        )
        return {
            "sequence": encoded,
            "escape_scores": escape_scores,
            "binding_energies": binding_energies,
            "total_escape": torch.tensor(self.safe_float(item.get("total_escape_score", 0.5)), dtype=torch.float32),
            "neutralization_resistance": torch.tensor(self.safe_float(item.get("neutralization_resistance", 0.5)), dtype=torch.float32),
        }


# =============================================================================
# CELL 5: Models
# =============================================================================
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


class AAVTropismLoss(nn.Module):
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
        total = tissue_loss + delivery_loss + 0.5 * immune_loss + 0.3 * tissue_score_loss
        return {"total": total, "tissue": tissue_loss, "delivery": delivery_loss,
                "immune": immune_loss, "tissue_scores": tissue_score_loss}


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
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, targets):
        delivery_loss = self.mse(pred["delivery_score"], targets["delivery_efficiency"])
        organoid_loss = self.mse(pred["organoid_counts"], targets["organoid_counts"])
        immune_loss = self.bce(pred["safety"][:, 0], targets["immune_activation"])
        cytotoxicity_loss = self.bce(pred["safety"][:, 1], 1.0 - targets["cytotoxicity"])
        total = delivery_loss + 0.5 * organoid_loss + 0.3 * (immune_loss + cytotoxicity_loss)
        return {"total": total, "delivery": delivery_loss, "organoid": organoid_loss,
                "immune": immune_loss, "cytotoxicity": cytotoxicity_loss}


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


# =============================================================================
# CELL 6: Training Functions
# =============================================================================
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


def train_aav(data, device, num_epochs=50, batch_size=256, lr=3e-4, ckpt_dir="/kaggle/working/checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    dataset = AAVSequenceDataset(data, max_len=50)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = AAVTropismTransformer().to(device)
    criterion = AAVTropismLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop = EarlyStopping(patience=10)
    scaler = torch.amp.GradScaler(device.type, enabled=USE_CUDA)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"AAV Tropism Transformer | {params:,} params")
    print(f"  Train: {train_size} | Val: {val_size} | Epochs: {num_epochs} | Batch: {batch_size}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            seqs = batch["sequence"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
            with torch.amp.autocast(device_type=device.type, enabled=USE_CUDA):
                pred = model(seqs)
                loss = criterion(pred, targets)["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            n += 1

        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                seqs = batch["sequence"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
                pred = model(seqs)
                val_loss += criterion(pred, targets)["total"].item()
                vn += 1
        val_loss /= max(vn, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        elapsed = time.time() - t0
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            marker = " *"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "loss": val_loss}, f"{ckpt_dir}/aav_tropism_best.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.6f} | {elapsed:.1f}s{marker}")

        early_stop.step(val_loss)
        if early_stop.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                 "loss": val_loss}, f"{ckpt_dir}/aav_tropism_final.pt")
    print(f"  Best val: {best_val:.4f}")
    return model, history


def train_lnp(data, device, num_epochs=100, batch_size=16, lr=5e-4, ckpt_dir="/kaggle/working/checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    dataset = LNPDeliveryDataset(data)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = LNPDeliveryMLP().to(device)
    criterion = LNPDeliveryLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop = EarlyStopping(patience=20)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"LNP Delivery MLP | {params:,} params")
    print(f"  Train: {train_size} | Val: {val_size} | Epochs: {num_epochs} | Batch: {batch_size}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            feats = batch["features"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "features"}
            pred = model(feats)
            loss = criterion(pred, targets)["total"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1

        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "features"}
                pred = model(feats)
                val_loss += criterion(pred, targets)["total"].item()
                vn += 1
        val_loss /= max(vn, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            marker = " *"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "loss": val_loss}, f"{ckpt_dir}/lnp_delivery_best.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s{marker}")

        early_stop.step(val_loss)
        if early_stop.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                 "loss": val_loss}, f"{ckpt_dir}/lnp_delivery_final.pt")
    print(f"  Best val: {best_val:.4f}")
    return model, history


def train_immune(data, device, num_epochs=50, batch_size=256, lr=3e-4, ckpt_dir="/kaggle/working/checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    dataset = ImmuneEscapeDataset(data, max_len=50)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = ImmuneEscapeTransformer().to(device)
    criterion = ImmuneEscapeLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stop = EarlyStopping(patience=10)
    scaler = torch.amp.GradScaler(device.type, enabled=USE_CUDA)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"Immune Escape Transformer | {params:,} params")
    print(f"  Train: {train_size} | Val: {val_size} | Epochs: {num_epochs} | Batch: {batch_size}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            seqs = batch["sequence"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
            with torch.amp.autocast(device_type=device.type, enabled=USE_CUDA):
                pred = model(seqs)
                loss = criterion(pred, targets)["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            n += 1

        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                seqs = batch["sequence"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
                pred = model(seqs)
                val_loss += criterion(pred, targets)["total"].item()
                vn += 1
        val_loss /= max(vn, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            marker = " *"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "loss": val_loss}, f"{ckpt_dir}/immune_escape_best.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s{marker}")

        early_stop.step(val_loss)
        if early_stop.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                 "loss": val_loss}, f"{ckpt_dir}/immune_escape_final.pt")
    print(f"  Best val: {best_val:.4f}")
    return model, history


# =============================================================================
# CELL 7: Visualization
# =============================================================================
def plot_history(history, title, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train", color="crimson", lw=2)
    ax.plot(history["val_loss"], label="Val", color="dodgerblue", lw=2, ls="--")
    ax.set_title(f"{title} - Loss Curve", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# =============================================================================
# CELL 8: Run Training
# =============================================================================
if not USE_CUDA:
    print("WARNING: Running on CPU. Training will be slow.")
    print("TIP: In Kaggle, set Runtime -> Accelerator -> GPU P100 or T4")

# Adjust params for CPU
aav_batch = 64 if USE_CUDA else 32
aav_epochs = 50 if USE_CUDA else 10
lnp_batch = 16
lnp_epochs = 100 if USE_CUDA else 20
imm_batch = 64 if USE_CUDA else 32
imm_epochs = 50 if USE_CUDA else 10

aav_model, aav_hist = train_aav(aav_data, device, num_epochs=aav_epochs, batch_size=aav_batch)
plot_history(aav_hist, "AAV Tropism", "/kaggle/working/aav_loss.png")

lnp_model, lnp_hist = train_lnp(lnp_data, device, num_epochs=lnp_epochs, batch_size=lnp_batch)
plot_history(lnp_hist, "LNP Delivery", "/kaggle/working/lnp_loss.png")

immune_model, immune_hist = train_immune(immune_data, device, num_epochs=imm_epochs, batch_size=imm_batch)
plot_history(immune_hist, "Immune Escape", "/kaggle/working/immune_loss.png")

# =============================================================================
# CELL 9: Summary & Download
# =============================================================================
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)

for pt in Path("/kaggle/working/checkpoints").glob("*.pt"):
    print(f"  {pt.name}: {pt.stat().st_size/1024/1024:.1f} MB")

for name, hist in [("aav", aav_hist), ("lnp", lnp_hist), ("immune", immune_hist)]:
    with open(f"/kaggle/working/{name}_history.json", "w") as f:
        json.dump(hist, f)

print("\nDownload from Kaggle: Output -> Download")
