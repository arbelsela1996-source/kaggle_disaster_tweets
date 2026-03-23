from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import config


def load_train_dataframe(path: str | None = None) -> pd.DataFrame:
    """Load the Kaggle train.csv file."""
    csv_path = path or config.train_file
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Train CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # Basic cleanup: ensure text column exists and fill missing values
    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError("Expected columns 'text' and 'target' in train.csv")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    return df


def train_val_split(
    df: pd.DataFrame, val_size: float | None = None, random_state: int | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and validation sets with stratification."""
    val_fraction = val_size if val_size is not None else config.val_size
    seed = random_state if random_state is not None else config.random_seed

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=df["target"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_and_split_train_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience helper: load train.csv and return (train_df, val_df)."""
    df = load_train_dataframe()
    return train_val_split(df)

