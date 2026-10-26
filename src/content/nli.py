"""Utilities for loading and evaluating Natural Language Inference models."""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS = ("entailment", "neutral", "contradiction")

def normalize_label(label: str) -> str:
    """Normalize a model label to entailment, neutral, or contradiction."""
    label = label.lower().replace("-", "_").replace(" ", "_")

    if "entail" in label:
        return "entailment"
    if "neutral" in label:
        return "neutral"
    if "contradict" in label:
        return "contradiction"

    raise ValueError(f"Unrecognized NLI label: {label!r}")


def load_nli_model(model_name: str, device: str | None = None):
    """Load an NLI tokenizer and sequence-classification model."""
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    label_mapping = {
        int(index): normalize_label(label)
        for index, label in model.config.id2label.items()
    }

    if set(label_mapping.values()) != set(LABELS):
        raise ValueError(
            f"Model labels do not cover {LABELS}: {label_mapping}"
        )

    return tokenizer, model, device, label_mapping


def predict_nli(
    tokenizer,
    model,
    pairs: list[tuple[str, str]],
    device: str,
    label_mapping: dict[int, str],
    batch_size: int = 8,
) -> pd.DataFrame:
    """Predict NLI probabilities for premise-hypothesis pairs."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if not pairs:
        return pd.DataFrame(
            columns=["entailment", "neutral", "contradiction"]
        )

    results = []

    model.eval()

    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            premises = [pair[0] for pair in batch]
            hypotheses = [pair[1] for pair in batch]

            inputs = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1).cpu()

            for row in probabilities:
                scores = {
                    label_mapping[index]: float(row[index])
                    for index in range(len(label_mapping))
                }
                results.append(scores)

    return pd.DataFrame(results, columns=LABELS)