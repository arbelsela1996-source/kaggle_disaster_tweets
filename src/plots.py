from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str | None = None,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_metric_curves(
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str,
    save_path: str | None = None,
) -> None:
    epochs = range(1, len(train_metrics) + 1)
    plt.figure()
    plt.plot(epochs, train_metrics, label=f"Train {metric_name}")
    plt.plot(epochs, val_metrics, label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Training and validation {metric_name}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Dict[int, str],
    save_path: str | None = None,
) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

