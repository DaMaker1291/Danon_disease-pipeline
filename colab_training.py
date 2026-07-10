"""
=============================================================================
LONGEVITY PIPELINE - GOOGLE COLAB TRAINING
=============================================================================
Train all 3 ML models (AAV tropism, LNP delivery, immune escape) on GPU.

INSTRUCTIONS:
  1. Open Google Colab (colab.research.google.com)
  2. Set runtime to GPU: Runtime -> Change runtime type -> T4 GPU
  3. Upload this file and the colab_data.zip, OR mount Google Drive
  4. Run all cells top-to-bottom

The notebook:
  - Installs all dependencies
  - Loads real data from Fit4Function (100K AAV) + LNPDB (773 LNP)
  - Trains 3 neural networks with GPU acceleration
  - Generates diagnostic plots
  - Saves trained checkpoints
=============================================================================
"""

# =============================================================================
# CELL 1: Setup & Install Dependencies
# =============================================================================
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install numpy matplotlib scikit-learn pandas -q

import os
import sys
import json
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# =============================================================================
# CELL 2: Upload Data
# =============================================================================
# Option A: Upload from local machine
try:
    from google.colab import files
    print("Upload colab_data.zip:")
    uploaded = files.upload()
    !unzip -o colab_data.zip -d /content/data/ 2>/dev/null
    DATA_DIR = "/content/data"
    print(f"Data loaded from upload: {DATA_DIR}")
except Exception as e:
    # Option B: Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_DIR = "/content/drive/MyDrive/longevity_data"
        print(f"Data loaded from Drive: {DATA_DIR}")
    except:
        DATA_DIR = "/content/data"
        print(f"Using default: {DATA_DIR}")

# =============================================================================
# CELL 3: Amino Acid Encoding
# =============================================================================
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

# =============================================================================
# CELL 4: Datasets
# =============================================================================
class AAVSequenceDataset(Dataset):
    def __init__(self, data_path, max_len=50):
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
        immune_escape = torch.tensor(
            item.get("immune_escape_score", item.get("total_escape_score", 0.5)),
            dtype=torch.float32
        )

        return {
            "sequence": encoded,
            "tissue_target": tissue_target,
            "tissue_scores": tissue_scores,
            "delivery_efficiency": delivery_eff,
            "immune_escape": immune_escape,
        }


class LNPDeliveryDataset(Dataset):
    def __init__(self, data_path):
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


class ImmuneEscapeDataset(Dataset):
    def __init__(self, data_path, max_len=50):
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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
        self._init_weights()

    def _init_weights(self):
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
    def __init__(self, tissue_weight=1.0, delivery_weight=1.0, immune_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.tissue_weight = tissue_weight
        self.delivery_weight = delivery_weight
        self.immune_weight = immune_weight

    def forward(self, predictions, targets):
        tissue_loss = self.ce_loss(predictions["tissue_logits"], targets["tissue_target"])
        delivery_loss = self.mse_loss(predictions["delivery_score"].squeeze(), targets["delivery_efficiency"])
        immune_score = predictions["immune_score"].squeeze().clamp(0.001, 0.999)
        immune_loss = self.bce_loss(immune_score, targets["immune_escape"])
        tissue_score_loss = self.mse_loss(predictions["tissue_scores"], targets["tissue_scores"])
        total = (self.tissue_weight * tissue_loss +
                 self.delivery_weight * delivery_loss +
                 self.immune_weight * immune_loss +
                 0.3 * tissue_score_loss)
        return {"total": total, "tissue": tissue_loss, "delivery": delivery_loss,
                "immune": immune_loss, "tissue_scores": tissue_score_loss}


class LNPDeliveryMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=1, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4),
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
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.delivery_weight = delivery_weight
        self.organoid_weight = organoid_weight
        self.safety_weight = safety_weight

    def forward(self, predictions, targets):
        delivery_loss = self.mse_loss(predictions["delivery_score"], targets["delivery_efficiency"])
        organoid_loss = self.mse_loss(predictions["organoid_counts"], targets["organoid_counts"])
        immune_loss = self.bce_loss(predictions["safety"][:, 0], targets["immune_activation"])
        cytotoxicity_loss = self.bce_loss(predictions["safety"][:, 1], 1.0 - targets["cytotoxicity"])
        total = (self.delivery_weight * delivery_loss +
                 self.organoid_weight * organoid_loss +
                 self.safety_weight * (immune_loss + cytotoxicity_loss))
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


def train_aav_model(data_path, device, num_epochs=30, batch_size=64,
                    learning_rate=3e-4, subsample=None, checkpoint_dir="/content/checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset = AAVSequenceDataset(data_path, max_len=50)
    if subsample and subsample < len(dataset):
        indices = np.random.choice(len(dataset), subsample, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info("Subsampled AAV to %d samples", subsample)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = AAVTropismTransformer().to(device)
    criterion = AAVTropismLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=10)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"AAV Tropism Transformer Training")
    print(f"  Device: {device}")
    print(f"  Train: {train_size} | Val: {val_size}")
    print(f"  Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            sequences = batch["sequence"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}

            with torch.amp.autocast(enabled=torch.cuda.is_available()):
                predictions = model(sequences)
                losses = criterion(predictions, targets)
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += losses["total"].item()
            n_batches += 1

        train_loss /= n_batches
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch["sequence"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
                predictions = model(sequences)
                losses = criterion(predictions, targets)
                val_loss += losses["total"].item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.6f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "loss": val_loss}, f"{checkpoint_dir}/aav_tropism_best.pt")

        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                 "loss": val_loss}, f"{checkpoint_dir}/aav_tropism_final.pt")
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, history


def train_lnp_model(data_path, device, num_epochs=50, batch_size=16,
                    learning_rate=3e-4, checkpoint_dir="/content/checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset = LNPDeliveryDataset(data_path)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = LNPDeliveryMLP().to(device)
    criterion = LNPDeliveryLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=15)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"LNP Delivery MLP Training")
    print(f"  Device: {device}")
    print(f"  Train: {train_size} | Val: {val_size}")
    print(f"  Epochs: {num_epochs} | Batch: {batch_size}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "features"}
            predictions = model(features)
            losses = criterion(predictions, targets)
            loss = losses["total"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "features"}
                predictions = model(features)
                losses = criterion(predictions, targets)
                val_loss += losses["total"].item()
                val_batches += 1
        val_loss /= max(val_batches, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "loss": val_loss}, f"{checkpoint_dir}/lnp_delivery_best.pt")

        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                 "loss": val_loss}, f"{checkpoint_dir}/lnp_delivery_final.pt")
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, history


def train_immune_model(data_path, device, num_epochs=30, batch_size=64,
                       learning_rate=3e-4, subsample=None, checkpoint_dir="/content/checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset = ImmuneEscapeDataset(data_path, max_len=50)
    if subsample and subsample < len(dataset):
        indices = np.random.choice(len(dataset), subsample, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info("Subsampled Immune to %d samples", subsample)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = ImmuneEscapeTransformer().to(device)
    criterion = ImmuneEscapeLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=10)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Immune Escape Transformer Training")
    print(f"  Device: {device}")
    print(f"  Train: {train_size} | Val: {val_size}")
    print(f"  Epochs: {num_epochs} | Batch: {batch_size}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            sequences = batch["sequence"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
            with torch.amp.autocast(enabled=torch.cuda.is_available()):
                predictions = model(sequences)
                losses = criterion(predictions, targets)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += losses["total"].item()
            n_batches += 1

        train_loss /= n_batches
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch["sequence"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "sequence"}
                predictions = model(sequences)
                losses = criterion(predictions, targets)
                val_loss += losses["total"].item()
                val_batches += 1
        val_loss /= max(val_batches, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "loss": val_loss}, f"{checkpoint_dir}/immune_escape_best.pt")

        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                 "loss": val_loss}, f"{checkpoint_dir}/immune_escape_final.pt")
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, history


# =============================================================================
# CELL 7: Visualization
# =============================================================================
def plot_training_history(history, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train", color="crimson", lw=2)
    axes[0].plot(history["val_loss"], label="Val", color="dodgerblue", lw=2, ls="--")
    axes[0].set_title(f"{title} - Loss Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, ls=":", alpha=0.5)

    if "lr" in history:
        axes[1].plot(history["lr"], color="forestgreen", lw=2)
        axes[1].set_title(f"{title} - Learning Rate", fontweight="bold")
    else:
        improvement = [(history["train_loss"][i] - history["train_loss"][i+1])
                       for i in range(len(history["train_loss"])-1)]
        axes[1].bar(range(len(improvement)), improvement,
                    color=["forestgreen" if v > 0 else "crimson" for v in improvement])
        axes[1].set_title(f"{title} - Loss Improvement per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# =============================================================================
# CELL 8: Run Training
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Train AAV Tropism ---
print("\n" + "="*60)
print("PHASE 1/3: AAV Tropism Model")
print("="*60)
aav_model, aav_history = train_aav_model(
    data_path=f"{DATA_DIR}/real_screening_aav_tropism.json",
    device=device,
    num_epochs=30,
    batch_size=128,
    learning_rate=3e-4,
    subsample=20000,
)
plot_training_history(aav_history, "AAV Tropism Transformer",
                      save_path="/content/diagnostics/aav_training.png")

# --- Train LNP Delivery ---
print("\n" + "="*60)
print("PHASE 2/3: LNP Delivery Model")
print("="*60)
lnp_model, lnp_history = train_lnp_model(
    data_path=f"{DATA_DIR}/real_screening_lnp_delivery.json",
    device=device,
    num_epochs=50,
    batch_size=16,
    learning_rate=5e-4,
)
plot_training_history(lnp_history, "LNP Delivery MLP",
                      save_path="/content/diagnostics/lnp_training.png")

# --- Train Immune Escape ---
print("\n" + "="*60)
print("PHASE 3/3: Immune Escape Model")
print("="*60)
immune_model, immune_history = train_immune_model(
    data_path=f"{DATA_DIR}/real_screening_immune_escape.json",
    device=device,
    num_epochs=30,
    batch_size=128,
    learning_rate=3e-4,
    subsample=20000,
)
plot_training_history(immune_history, "Immune Escape Transformer",
                      save_path="/content/diagnostics/immune_training.png")

# =============================================================================
# CELL 9: Save & Download
# =============================================================================
print("\n" + "="*60)
print("TRAINING COMPLETE - SAVING CHECKPOINTS")
print("="*60)

os.makedirs("/content/diagnostics", exist_ok=True)
for f in Path("/content/checkpoints").glob("*.pt"):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  {f.name}: {size_mb:.1f} MB")

# Save training summaries
for name, history in [("aav", aav_history), ("lnp", lnp_history), ("immune", immune_history)]:
    with open(f"/content/diagnostics/{name}_training_summary.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved {name} training summary")

print("\nDownload checkpoints:")
try:
    from google.colab import files
    for pt in Path("/content/checkpoints").glob("*.pt"):
        files.download(str(pt))
    for json_f in Path("/content/diagnostics").glob("*_summary.json"):
        files.download(str(json_f))
except Exception as e:
    print(f"  Auto-download failed: {e}")
    print("  Use: from google.colab import files; files.download('/content/checkpoints/aav_tropism_best.pt')")

# =============================================================================
# CELL 10: Quick Inference Demo
# =============================================================================
print("\n" + "="*60)
print("INFERENCE DEMO")
print("="*60)

aav_model.eval()
lnp_model.eval()
immune_model.eval()

# Test AAV
test_seq = torch.randint(0, 20, (1, 50), device=device)
with torch.no_grad():
    aav_pred = aav_model(test_seq)
print(f"AAV prediction - tissue logits: {aav_pred['tissue_logits'].softmax(1).cpu().numpy().round(3)}")
print(f"  delivery: {aav_pred['delivery_score'].item():.4f}")
print(f"  immune: {aav_pred['immune_score'].item():.4f}")

# Test LNP
test_features = torch.rand(1, 9, device=device)
with torch.no_grad():
    lnp_pred = lnp_model(test_features)
print(f"LNP prediction - delivery: {lnp_pred['delivery_score'].item():.4f}")
print(f"  safety: {lnp_pred['safety'].cpu().numpy().round(3)}")

# Test Immune
test_seq2 = torch.randint(0, 20, (1, 50), device=device)
with torch.no_grad():
    imm_pred = immune_model(test_seq2)
print(f"Immune prediction - total_escape: {imm_pred['total_escape'].item():.4f}")
print(f"  resistance: {imm_pred['resistance'].item():.4f}")

print("\nDone! All 3 models trained on real data with GPU acceleration.")
