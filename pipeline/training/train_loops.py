import os
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pipeline.models.architectures import (
    AAVTropismTransformer, AAVTropismLoss, AAVSequenceDataset,
    LNPDeliveryMLP, LNPDeliveryLoss, LNPDeliveryDataset,
    ImmuneEscapeTransformer, ImmuneEscapeLoss, ImmuneEscapeDataset,
)
from pipeline.training.telemetry import PrecisionTelemetryLogger

logger = logging.getLogger(__name__)


class TrainingConfig:
    def __init__(self):
        self.batch_size = 64
        self.num_epochs = 50
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.warmup_steps = 500
        self.max_grad_norm = 1.0
        self.patience = 10
        self.checkpoint_dir = "./checkpoints"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        self.gradient_accumulation_steps = 4
        self.fp16 = torch.cuda.is_available()


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


class AAVTrainer:
    def __init__(self, config: TrainingConfig = None, diagnostics_dir: str = "./diagnostics"):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.telemetry = PrecisionTelemetryLogger(os.path.join(diagnostics_dir, "aav"))

    def train(self, data_path: str, val_split: float = 0.1):
        logger.info("Starting AAV Tropism Model Training")
        logger.info("Device: %s | Batch size: %d | Epochs: %d",
                     self.device, self.config.batch_size, self.config.num_epochs)

        dataset = AAVSequenceDataset(data_path)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=True
        )

        model = AAVTropismTransformer().to(self.device)
        criterion = AAVTropismLoss()
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate,
                         weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        early_stopping = EarlyStopping(patience=self.config.patience)

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                sequences = batch["sequence"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "sequence"}

                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    predictions = model(sequences)
                    losses = criterion(predictions, targets)
                    loss = losses["total"] / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += losses["total"].item()
                num_batches += 1

            train_loss /= num_batches
            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    sequences = batch["sequence"].to(self.device)
                    targets = {k: v.to(self.device) for k, v in batch.items() if k != "sequence"}
                    predictions = model(sequences)
                    losses = criterion(predictions, targets)
                    val_loss += losses["total"].item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            self.telemetry.log_step(train_loss, val_loss, current_lr, grad_norm)
            self.telemetry.log_weight_stats(epoch, model)

            logger.info("Epoch %d/%d | Train: %.4f | Val: %.4f | LR: %.6f | GradNorm: %.4f",
                         epoch + 1, self.config.num_epochs, train_loss, val_loss,
                         current_lr, grad_norm)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model, optimizer, epoch, val_loss, "aav_tropism_best")

            early_stopping.step(val_loss)
            if early_stopping.should_stop:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        self._save_checkpoint(model, optimizer, epoch, val_loss, "aav_tropism_final")
        self._save_history(history, "aav_tropism_history")

        self.telemetry.generate_all_diagnostics(model)
        self.telemetry.export_summary_json("aav_training_summary.json")

        return model, history

    def _save_checkpoint(self, model, optimizer, epoch, loss, name):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "loss": loss,
        }, path)
        logger.info("Checkpoint saved: %s", path)

    def _save_history(self, history, name):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(history, f)


class LNPTrainer:
    def __init__(self, config: TrainingConfig = None, diagnostics_dir: str = "./diagnostics"):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.telemetry = PrecisionTelemetryLogger(os.path.join(diagnostics_dir, "lnp"))

    def train(self, data_path: str, val_split: float = 0.1):
        logger.info("Starting LNP Delivery Model Training")

        dataset = LNPDeliveryDataset(data_path)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers
        )

        model = LNPDeliveryMLP().to(self.device)
        criterion = LNPDeliveryLoss()
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate,
                         weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        early_stopping = EarlyStopping(patience=self.config.patience)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                features = batch["features"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "features"}

                predictions = model(features)
                losses = criterion(predictions, targets)
                loss = losses["total"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches
            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(self.device)
                    targets = {k: v.to(self.device) for k, v in batch.items() if k != "features"}
                    predictions = model(features)
                    losses = criterion(predictions, targets)
                    val_loss += losses["total"].item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            self.telemetry.log_step(train_loss, val_loss, current_lr, grad_norm)
            self.telemetry.log_weight_stats(epoch, model)

            logger.info("Epoch %d/%d | Train: %.4f | Val: %.4f | GradNorm: %.4f",
                         epoch + 1, self.config.num_epochs, train_loss, val_loss, grad_norm)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model, optimizer, epoch, val_loss, "lnp_delivery_best")

            early_stopping.step(val_loss)
            if early_stopping.should_stop:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        self._save_checkpoint(model, optimizer, epoch, val_loss, "lnp_delivery_final")
        self._save_history(history, "lnp_delivery_history")

        self.telemetry.generate_all_diagnostics(model)
        self.telemetry.export_summary_json("lnp_training_summary.json")

        return model, history

    def _save_checkpoint(self, model, optimizer, epoch, loss, name):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "loss": loss,
        }, path)

    def _save_history(self, history, name):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(history, f)


class ImmuneTrainer:
    def __init__(self, config: TrainingConfig = None, diagnostics_dir: str = "./diagnostics"):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.telemetry = PrecisionTelemetryLogger(os.path.join(diagnostics_dir, "immune"))

    def train(self, data_path: str, val_split: float = 0.1):
        logger.info("Starting Immune Escape Model Training")

        dataset = ImmuneEscapeDataset(data_path)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=True
        )

        model = ImmuneEscapeTransformer().to(self.device)
        criterion = ImmuneEscapeLoss()
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate,
                         weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        early_stopping = EarlyStopping(patience=self.config.patience)

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                sequences = batch["sequence"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "sequence"}

                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    predictions = model(sequences)
                    losses = criterion(predictions, targets)
                    loss = losses["total"] / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += losses["total"].item()
                num_batches += 1

            train_loss /= num_batches
            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    sequences = batch["sequence"].to(self.device)
                    targets = {k: v.to(self.device) for k, v in batch.items() if k != "sequence"}
                    predictions = model(sequences)
                    losses = criterion(predictions, targets)
                    val_loss += losses["total"].item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            self.telemetry.log_step(train_loss, val_loss, current_lr, grad_norm)
            self.telemetry.log_weight_stats(epoch, model)

            logger.info("Epoch %d/%d | Train: %.4f | Val: %.4f | GradNorm: %.4f",
                         epoch + 1, self.config.num_epochs, train_loss, val_loss, grad_norm)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model, optimizer, epoch, val_loss, "immune_escape_best")

            early_stopping.step(val_loss)
            if early_stopping.should_stop:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        self._save_checkpoint(model, optimizer, epoch, val_loss, "immune_escape_final")
        self._save_history(history, "immune_escape_history")

        self.telemetry.generate_all_diagnostics(model)
        self.telemetry.export_summary_json("immune_training_summary.json")

        return model, history

    def _save_checkpoint(self, model, optimizer, epoch, loss, name):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "loss": loss,
        }, path)

    def _save_history(self, history, name):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(history, f)


class ModelManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_aav_model(self, checkpoint_name: str = "aav_tropism_best") -> AAVTropismTransformer:
        model = AAVTropismTransformer().to(self.device)
        path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded AAV model from %s (loss: %.4f)", path, checkpoint.get("loss", 0))
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded AAV model from %s (state dict only)", path)
        else:
            logger.warning("No checkpoint found at %s, using random weights", path)
        model.eval()
        return model

    def load_lnp_model(self, checkpoint_name: str = "lnp_delivery_best") -> LNPDeliveryMLP:
        model = LNPDeliveryMLP().to(self.device)
        path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded LNP model from %s (loss: %.4f)", path, checkpoint.get("loss", 0))
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded LNP model from %s (state dict only)", path)
        else:
            logger.warning("No checkpoint found at %s, using random weights", path)
        model.eval()
        return model

    def load_immune_model(self, checkpoint_name: str = "immune_escape_best") -> ImmuneEscapeTransformer:
        model = ImmuneEscapeTransformer().to(self.device)
        path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded Immune model from %s (loss: %.4f)", path, checkpoint.get("loss", 0))
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded Immune model from %s (state dict only)", path)
        else:
            logger.warning("No checkpoint found at %s, using random weights", path)
        model.eval()
        return model

    def export_onnx(self, model, name: str, input_shape: tuple):
        path = os.path.join(self.checkpoint_dir, f"{name}.onnx")
        dummy = torch.zeros(input_shape, device=self.device)
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"])
        logger.info("Exported ONNX model: %s", path)
        return path
