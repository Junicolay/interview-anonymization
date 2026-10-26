"""Utilities for sentence-embedding evaluation."""

import numpy as np
import pandas as pd


def evaluate_text_pairs(model, pairs_by_category):
    """Encode text pairs and calculate cosine similarities.

    Args:
        model: A loaded SentenceTransformer model.
        pairs_by_category: Dictionary mapping category names to lists
            of (sentence_1, sentence_2) tuples.

    Returns:
        DataFrame with category, sentence_1, sentence_2, and cosine.
    """
    rows = []

    for category, pairs in pairs_by_category.items():
        if not pairs:
            continue

        sentences = [sentence for pair in pairs for sentence in pair]

        embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
            raise ValueError(
                f"Unexpected embedding shape: {embeddings.shape}"
            )

        for i, (sentence_1, sentence_2) in enumerate(pairs):
            cosine = float(
                np.dot(embeddings[2 * i], embeddings[2 * i + 1])
            )

            rows.append({
                "category": category,
                "sentence_1": sentence_1,
                "sentence_2": sentence_2,
                "cosine": cosine,
            })

    return pd.DataFrame(
        rows,
        columns=["category", "sentence_1", "sentence_2", "cosine"],
    )
