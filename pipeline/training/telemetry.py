import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class PrecisionTelemetryLogger:
    def __init__(self, output_dir="./diagnostics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.weight_stats = []

    def log_step(self, train_loss, val_loss=None, lr=None, grad_norm=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)

    def log_weight_stats(self, epoch, model):
        stats = {"epoch": epoch}
        for name, param in model.named_parameters():
            if param.requires_grad:
                w = param.detach().cpu().numpy().flatten()
                stats[name] = {
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                    "norm": float(np.linalg.norm(w)),
                }
        self.weight_stats.append(stats)

    def generate_loss_plot(self, filename="loss_curve.png"):
        if not self.train_losses:
            return None

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(self.train_losses, label="Training Loss", color="crimson", lw=2)
        if self.val_losses:
            ax1.plot(self.val_losses, label="Validation Loss", color="dodgerblue",
                     linestyle="--", lw=2)

        ax1.set_title("Pipeline Loss Optimization Landscape", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Training Steps", fontsize=10)
        ax1.set_ylabel("Loss Value", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        if self.learning_rates:
            ax2 = ax1.twinx()
            ax2.plot(self.learning_rates, label="Learning Rate", color="forestgreen",
                     linestyle=":", lw=1.5, alpha=0.7)
            ax2.set_ylabel("Learning Rate", fontsize=10)
            ax2.legend(loc="upper left")

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Telemetry] Saved loss landscape to: {path}")
        return path

    def generate_weight_distribution(self, model, filename="weights_distribution.png"):
        weights = []
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                weights.extend(param.detach().cpu().numpy().flatten())

        if not weights:
            return None

        weights = np.array(weights)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(weights, bins=100, color="teal", alpha=0.75, edgecolor="black", linewidth=0.2)
        axes[0].set_title("Layer Weights Distribution", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Weight Values", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].grid(True, linestyle=":", alpha=0.6)

        mean_w = np.mean(weights)
        std_w = np.std(weights)
        skew_w = float(np.mean(((weights - mean_w) / max(std_w, 1e-8)) ** 3))
        kurt_w = float(np.mean(((weights - mean_w) / max(std_w, 1e-8)) ** 4) - 3)

        stats_text = (
            f"Mean: {mean_w:.4f}\n"
            f"Std: {std_w:.4f}\n"
            f"Skewness: {skew_w:.4f}\n"
            f"Kurtosis: {kurt_w:.4f}\n"
            f"Min: {np.min(weights):.4f}\n"
            f"Max: {np.max(weights):.4f}\n"
            f"L2 Norm: {np.linalg.norm(weights):.2f}"
        )
        axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes,
                     fontsize=11, verticalalignment="center", fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        axes[1].set_title("Weight Statistics", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Telemetry] Saved weight distribution to: {path}")
        return path

    def generate_gradient_norm_plot(self, filename="gradient_norms.png"):
        if not self.gradient_norms:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.gradient_norms, color="darkorange", lw=2)
        ax.set_title("Gradient Norm per Training Step", fontsize=12, fontweight="bold")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("L2 Gradient Norm", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

        mean_gn = np.mean(self.gradient_norms)
        ax.axhline(y=mean_gn, color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {mean_gn:.4f}")
        ax.legend()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Telemetry] Saved gradient norms to: {path}")
        return path

    def generate_weight_evolution_plot(self, filename="weight_evolution.png"):
        if not self.weight_stats:
            return None

        param_names = [k for k in self.weight_stats[0].keys() if k != "epoch"]
        n_params = min(len(param_names), 6)
        selected = param_names[:n_params]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, pname in enumerate(selected):
            means = [s[pname]["mean"] for s in self.weight_stats if pname in s]
            stds = [s[pname]["std"] for s in self.weight_stats if pname in s]
            epochs = [s["epoch"] for s in self.weight_stats if pname in s]

            axes[idx].errorbar(epochs, means, yerr=stds, capsize=3, lw=2)
            axes[idx].set_title(pname.split(".")[-1], fontsize=9, fontweight="bold")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Value")
            axes[idx].grid(True, linestyle=":", alpha=0.5)

        for idx in range(n_params, 6):
            axes[idx].axis("off")

        fig.suptitle("Weight Evolution Across Epochs", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Telemetry] Saved weight evolution to: {path}")
        return path

    def generate_all_diagnostics(self, model=None):
        paths = []
        p = self.generate_loss_plot()
        if p:
            paths.append(p)
        p = self.generate_gradient_norm_plot()
        if p:
            paths.append(p)
        p = self.generate_weight_evolution_plot()
        if p:
            paths.append(p)
        if model is not None:
            p = self.generate_weight_distribution(model)
            if p:
                paths.append(p)
        return paths

    def export_summary_json(self, filename="training_summary.json"):
        import json
        summary = {
            "total_steps": len(self.train_losses),
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "loss_history": self.train_losses,
            "val_loss_history": self.val_losses,
            "gradient_norm_history": self.gradient_norms,
        }
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Telemetry] Saved summary to: {path}")
        return path
