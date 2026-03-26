from __future__ import annotations

import os
from pathlib import Path

from .config import config


def resolve_train_csv_path() -> str:
    """Prefer Kaggle competition mounts, then local data/train.csv."""
    kaggle_candidates = [
        "/kaggle/input/competitions/nlp-getting-started/train.csv",
        "/kaggle/input/nlp-getting-started/train.csv",
        "/kaggle/input/natural-language-processing-with-disaster-tweets/train.csv",
    ]
    for path in kaggle_candidates:
        if Path(path).is_file():
            return path
    if Path(config.train_file).is_file():
        return config.train_file
    raise FileNotFoundError(
        f"train.csv not found. Tried Kaggle paths and {config.train_file}"
    )


def resolve_test_csv_path() -> str:
    """Prefer Kaggle competition mounts, then local data/test.csv."""
    kaggle_candidates = [
        "/kaggle/input/competitions/nlp-getting-started/test.csv",
        "/kaggle/input/nlp-getting-started/test.csv",
        "/kaggle/input/natural-language-processing-with-disaster-tweets/test.csv",
    ]
    for path in kaggle_candidates:
        if Path(path).is_file():
            return path
    if Path(config.test_file).is_file():
        return config.test_file
    raise FileNotFoundError(
        f"test.csv not found. Tried Kaggle paths and {config.test_file}"
    )


def submission_output_paths() -> list[str]:
    """Paths to write submission.csv (project outputs + Kaggle working when present)."""
    paths = [os.path.join(config.outputs_dir, "submission.csv")]
    if Path("/kaggle/working").is_dir():
        paths.append("/kaggle/working/submission.csv")
    return paths
