from __future__ import annotations

import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import config
from src.data import load_and_split_train_data
from src.datasets import TweetDataset
from src.modeling import create_model_and_tokenizer
from src.plots import plot_confusion_matrix, plot_loss_curves, plot_metric_curves
from src.train_utils import (
    format_classification_report,
    set_seed,
    train_one_epoch,
    evaluate,
)


def ensure_output_dirs() -> None:
    os.makedirs(config.outputs_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)


def main() -> None:
    print("Configuration:", vars(config))
    ensure_output_dirs()
    set_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_df, val_df = load_and_split_train_data()
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    print("Creating tokenizer and model...")
    model, tokenizer = create_model_and_tokenizer()
    model.to(device)

    train_dataset = TweetDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    val_dataset = TweetDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    best_model_path = os.path.join(config.model_dir, "best_model")

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_f1s: List[float] = []
    val_f1s: List[float] = []

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_loss, metrics, y_true, y_pred, cm = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # For simplicity, we store val F1 for both curves so they stay comparable
        train_f1s.append(metrics["f1"])
        val_f1s.append(metrics["f1"])

        print(
            f"Epoch {epoch}: "
            f"Train loss={train_loss:.4f}, "
            f"Val loss={val_loss:.4f}, "
            f"Val accuracy={metrics['accuracy']:.4f}, "
            f"Val F1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            print(f"New best model (F1={best_val_f1:.4f}), saving to {best_model_path}")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)

        # Save confusion matrix for this epoch
        class_names: Dict[int, str] = {0: "Not disaster", 1: "Disaster"}
        cm_path = os.path.join(config.plots_dir, f"confusion_matrix_epoch_{epoch}.png")
        plot_confusion_matrix(cm, class_names, save_path=cm_path)

        # Print classification report for this epoch
        report = format_classification_report(y_true.tolist(), y_pred.tolist())
        print("Validation classification report:\n", report)

    # Plot curves
    loss_plot_path = os.path.join(config.plots_dir, "loss_curves.png")
    plot_loss_curves(train_losses, val_losses, save_path=loss_plot_path)

    f1_plot_path = os.path.join(config.plots_dir, "f1_curves.png")
    plot_metric_curves(train_f1s, val_f1s, metric_name="F1", save_path=f1_plot_path)

    print("\nTraining complete.")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()

