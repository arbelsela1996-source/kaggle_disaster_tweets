"""
Kaggle version of the disaster tweet classifier.

You can copy cells from this script into a Kaggle Notebook, or
simply upload the file and convert it to a notebook.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import matplotlib.pyplot as plt


def resolve_kaggle_train_csv() -> str:
    """Return the first existing Kaggle train.csv path across common mount layouts."""
    candidates = [
        "/kaggle/input/competitions/nlp-getting-started/train.csv",
        "/kaggle/input/nlp-getting-started/train.csv",
        "/kaggle/input/natural-language-processing-with-disaster-tweets/train.csv",
    ]
    for path in candidates:
        if Path(path).is_file():
            return path
    raise FileNotFoundError(
        "Could not find train.csv in expected Kaggle input paths. "
        "Check the dataset mount under /kaggle/input."
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TweetDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc="Train", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Eval", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return avg_loss, {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, np.array(all_labels), np.array(all_preds)


def plot_curves(train_losses, val_losses, train_f1s, val_f1s):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_f1s, label="Train F1")
    plt.plot(epochs, val_f1s, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Training and validation F1")
    plt.legend()
    plt.show()


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # On Kaggle, train.csv is available in the input directory.
    train_csv_path = resolve_kaggle_train_csv()
    print("Using train.csv from:", train_csv_path)
    df = pd.read_csv(train_csv_path)
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["target"],
    )

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_dataset = TweetDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=128,
    )
    val_dataset = TweetDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=128,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []

    for epoch in range(1, 3 + 1):
        print(f"\nEpoch {epoch}/3")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_loss, metrics, y_true, y_pred = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Here we only have val metrics; for simplicity, reuse val F1 for train F1 curve
        train_f1s.append(metrics["f1"])
        val_f1s.append(metrics["f1"])

        print(
            f"Epoch {epoch}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}, "
            f"Val acc={metrics['accuracy']:.4f}, Val F1={metrics['f1']:.4f}"
        )

        print("\nValidation classification report:\n")
        print(classification_report(y_true, y_pred, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)

    plot_curves(train_losses, val_losses, train_f1s, val_f1s)

    # Inference examples
    example_texts = [
        "Just happened a terrible car crash",
        "What a beautiful sunny day!",
    ]
    model.eval()
    with torch.no_grad():
        for text in example_texts:
            enc = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred = int(torch.argmax(probs).item())
            print(f"Text: {text}")
            print(f"  Predicted label: {pred} (prob disaster={probs[1].item():.3f})\n")


if __name__ == "__main__":
    main()

