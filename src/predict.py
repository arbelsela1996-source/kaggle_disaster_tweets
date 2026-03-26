from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from .datasets import TweetDataset


def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer


def predict_texts(
    texts: List[str],
    model,
    tokenizer,
    max_length: int = 128,
    device: torch.device | None = None,
) -> Tuple[List[int], List[float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_positive = probs[0, 1].item()
            pred_label = int(torch.argmax(logits, dim=-1).item())

            all_preds.append(pred_label)
            all_probs.append(prob_positive)

    return all_preds, all_probs


def predict_disaster_positive_probs(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return shape (N,) probabilities P(class=1) for each text."""
    dataset = TweetDataset(
        texts=texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    out: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            out.extend(probs.cpu().numpy().tolist())
    return np.array(out, dtype=np.float32)

