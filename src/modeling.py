from __future__ import annotations

from typing import Tuple

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import config


def create_tokenizer(model_name: str | None = None) -> PreTrainedTokenizerBase:
    name = model_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def create_model(
    model_name: str | None = None,
    num_labels: int = 2,
) -> PreTrainedModel:
    name = model_name or config.model_name
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=num_labels,
    )
    return model


def create_model_and_tokenizer(
    model_name: str | None = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    name = model_name or config.model_name
    tokenizer = create_tokenizer(name)
    model = create_model(name)
    return model, tokenizer

