"""
5-fold stratified CV with the E1 recipe (bert-base-uncased, 5 epochs, lr=2e-5, max_length=128).
Saves one checkpoint per fold, then averages disaster probabilities on test.csv and writes submission.csv.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src.config import config
from src.data import load_train_dataframe
from src.paths import resolve_test_csv_path, resolve_train_csv_path, submission_output_paths
from src.predict import load_model_and_tokenizer, predict_disaster_positive_probs
from train_disaster_bert import ExperimentSpec, train_single_split


# E1 recipe (matches benchmark E1_epochs_5)
KFOLD_E1 = ExperimentSpec(
    name="KFold_E1_5fold",
    description="Stratified 5-fold CV with bert-base E1 hyperparameters",
    model_name="bert-base-uncased",
    learning_rate=2e-5,
    num_epochs=5,
    max_length=128,
    seeds=[42],
)


def ensure_dirs() -> None:
    os.makedirs(config.outputs_dir, exist_ok=True)


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = resolve_train_csv_path()
    print(f"Train CSV: {train_path}")
    df = load_train_dataframe(train_path)
    print(f"Loaded {len(df)} labeled rows")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    texts = df["text"].values
    labels = df["target"].values

    kfold_root = os.path.join(config.outputs_dir, "kfold_e1")
    os.makedirs(kfold_root, exist_ok=True)

    fold_macro_f1: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val = df.iloc[val_idx].reset_index(drop=True)
        fold_dir = os.path.join(kfold_root, f"fold_{fold_idx}")
        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{n_splits} — train={len(fold_train)} val={len(fold_val)}")
        print("=" * 80)

        metrics = train_single_split(
            fold_train,
            fold_val,
            KFOLD_E1,
            seed=42,
            device=device,
            output_dir=os.path.join(fold_dir, "run"),
            save_artifacts=True,
        )
        fold_macro_f1.append(metrics["f1_macro"])
        print(
            f"[Fold {fold_idx}] val macro-F1={metrics['f1_macro']:.4f}, "
            f"acc={metrics['accuracy']:.4f}"
        )

    arr = np.array(fold_macro_f1, dtype=float)
    print("\n" + "#" * 80)
    print("K-FOLD SUMMARY (E1 recipe)")
    print("#" * 80)
    print(f"Per-fold macro-F1: {[round(x, 4) for x in fold_macro_f1]}")
    print(f"Mean macro-F1: {arr.mean():.4f} (+/- {arr.std():.4f})")

    # --- Test prediction: average probabilities across folds ---
    test_path = resolve_test_csv_path()
    print(f"\nTest CSV: {test_path}")
    test_df = pd.read_csv(test_path)
    if "text" not in test_df.columns or "id" not in test_df.columns:
        raise ValueError("test.csv must contain columns 'id' and 'text'")
    test_texts = test_df["text"].fillna("").astype(str).str.strip().tolist()

    all_fold_probs: List[np.ndarray] = []
    for fold_idx in range(1, n_splits + 1):
        model_dir = os.path.join(kfold_root, f"fold_{fold_idx}", "run", "best_model")
        print(f"Loading fold {fold_idx} model from {model_dir}")
        model, tokenizer = load_model_and_tokenizer(model_dir)
        model.to(device)
        probs = predict_disaster_positive_probs(
            test_texts,
            model,
            tokenizer,
            max_length=KFOLD_E1.max_length,
            batch_size=config.batch_size,
            device=device,
        )
        all_fold_probs.append(probs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_probs = np.stack(all_fold_probs, axis=0).mean(axis=0)
    preds = (avg_probs >= 0.5).astype(int)

    submission = pd.DataFrame({"id": test_df["id"], "target": preds})

    for out_path in submission_output_paths():
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        submission.to_csv(out_path, index=False)
        print(f"Wrote submission: {out_path}")

    print("\nDone. Submit the CSV on the Kaggle competition page.")


if __name__ == "__main__":
    main()
